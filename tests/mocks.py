"""Offline stand-ins for the two things the harness talks to.

`ScriptedProvider` answers `complete()` from rules keyed on the system prompt,
so one provider object can play the agent, the reviewer, and the reflection step
in the same test. `FakeMemcoClient` keeps memories in a list and returns the
same dataclasses the real client does.

Nothing here touches the network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from memco_harness.memco_client import (
    DataSource,
    FeedbackEntry,
    FeedbackResult,
    Insight,
    Memory,
    MemorySession,
    SearchResult,
    Tag,
    WriteResult,
)
from memco_harness.providers import Completion, Message, ToolDef, Usage

# Substrings that identify which component is calling, taken from the first
# line of each system prompt.
AGENT = "order-desk assistant"
REVIEWER = "order-desk supervisor"
LESSONS = "shared memory"
FEEDBACK = "grading how useful"

Responder = Callable[[str, list[Message]], "Completion | str"]


@dataclass
class ScriptedProvider:
    """A provider whose answers are chosen by matching the system prompt.

    Each rule is `(substring, response)`. A response may be a string, a
    `Completion`, or a callable taking `(system, messages)`.
    """

    rules: tuple[tuple[str, Any], ...] = ()
    default: Any = ""
    model: str = "scripted-model"
    spec: str = "scripted:scripted-model"
    usage: Usage = Usage()
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Completion:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        # A token count the runner can difference, so the usage plumbing is
        # exercised offline rather than only against a live provider.
        self.usage = self.usage + Usage(input_tokens=10, output_tokens=5, calls=1)
        for needle, response in self.rules:
            if needle in system:
                return _as_completion(response, system, messages)
        return _as_completion(self.default, system, messages)

    def calls_matching(self, needle: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if needle in call["system"]]


def _as_completion(response: Any, system: str, messages: list[Message]) -> Completion:
    if callable(response):
        response = response(system, messages)
    if isinstance(response, Completion):
        return response
    return Completion(text=str(response))


@dataclass
class StoredMemory:
    title: str
    content: str
    query: str
    source: str


MAX_HITS = 5


@dataclass
class FakeMemcoClient:
    """An in-memory Memco Shared Memory.

    Search is a plain word-overlap match, which is enough to show retrieval
    working in a test without pretending to be a search engine.

    It does model the one server behaviour the harness now leans on: within a
    session, a memory an earlier search already returned comes back as a
    reference carrying no insights. Leaving that out would let the episode path
    pass here while double-counting retrieval against a real server.
    """

    memories: list[StoredMemory] = field(default_factory=list)
    feedback: list[tuple[str, FeedbackEntry]] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    written: list[tuple[str | None, StoredMemory]] = field(default_factory=list)
    fail_next_search: bool = False
    fail_next_session: bool = False
    _session_counter: int = 0
    # Session id to the memories it has already been shown.
    _served: dict[str, set[str]] = field(default_factory=dict)

    def _mint_session(self) -> str:
        self._session_counter += 1
        session_id = f"session-{self._session_counter}"
        self.sessions.append(session_id)
        return session_id

    def open_session(self) -> MemorySession:
        if self.fail_next_session:
            self.fail_next_session = False
            return MemorySession(
                client=self, error="memco start_session failed: injected failure"
            )
        return MemorySession(client=self, session_id=self._mint_session())

    def search(
        self,
        query: str,
        tags: list[Tag] | None = None,
        session_id: str | None = None,
    ) -> SearchResult:
        self.searches.append(query)
        if self.fail_next_search:
            self.fail_next_search = False
            return SearchResult(session_id=None, error="memco search failed: injected failure")

        # No session named means a throwaway one for this call, which is what
        # the server does and what `report.reconcile` relies on.
        session_id = session_id or self._mint_session()
        served = self._served.setdefault(session_id, set())
        wanted = _words(query)

        memories: list[Memory] = []
        insights: list[Insight] = []
        for index, stored in enumerate(self.memories):
            if len(memories) >= MAX_HITS:
                break
            if not wanted & _words(f"{stored.title} {stored.content} {stored.query}"):
                continue
            idx = f"memory-{index}"
            if idx in served:
                memories.append(Memory(idx=idx, insights=()))
                continue
            served.add(idx)
            insight = Insight(
                idx=f"{idx}-insight-0",
                memory_idx=idx,
                title=stored.title,
                content=stored.content,
            )
            memories.append(Memory(idx=idx, insights=(insight,)))
            insights.append(insight)

        return SearchResult(
            session_id=session_id,
            memories=tuple(memories),
            insights=tuple(insights),
        )

    def create_memory(
        self,
        query: str,
        title: str,
        content: str,
        tags: list[Tag] | None = None,
        source: DataSource = DataSource.AGENT,
        session_id: str | None = None,
    ) -> WriteResult:
        stored = StoredMemory(title=title, content=content, query=query, source=source.name.lower())
        self.memories.append(stored)
        self.written.append((session_id, stored))
        return WriteResult(op_id=f"mem-{len(self.memories):04d}")

    def share_feedback(self, session_id: str, entries: list[FeedbackEntry]) -> FeedbackResult:
        for entry in entries:
            self.feedback.append((session_id, entry))
        return FeedbackResult(detail=f"recorded {len(entries)}")

    def close(self) -> None:
        """Nothing to close. Present so the runner can own a fake's lifecycle."""


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "to",
    "of",
    "on",
    "in",
    "is",
    "it",
    "we",
    "our",
    "this",
    "that",
    "with",
    "customer",
    "order",
    "asks",
    "about",
}


def _words(text: str) -> set[str]:
    return {
        word.strip(".,;:!?'\"()").lower()
        for word in text.split()
        if len(word) > 3 and word.lower() not in _STOPWORDS
    }
