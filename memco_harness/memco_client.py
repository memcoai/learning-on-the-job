"""A thin client for the Memco Memory MCP.

Exactly what the harness needs and nothing more: search memory, write a lesson,
and give feedback on what a search returned. The transport is the official MCP
Python SDK (2.x) over streamable HTTP; when the Memco SDK is published this
module is the only one that changes, so its public surface is kept small.

Calls are scoped to a domain, set once on the client and sent with every tool
call. The shipped scenario uses `knowledge`; override with MEMCO_MCP_DOMAIN.

Every method returns a result object. Memory calls never raise into the run: a
failure is recorded on the result, written to the task's line in the JSONL, and
the run continues. One retry is made for transient failures.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "DEFAULT_DOMAIN",
    "TOPIC",
    "TOPIC_TAG",
    "FeedbackEntry",
    "FeedbackResult",
    "Insight",
    "MemcoClient",
    "Memory",
    "SearchResult",
    "WriteResult",
    "build_client",
    "stamp",
]


def stamp() -> str:
    """Wall-clock instant, for measuring how long a write takes to become searchable.

    This is instrumentation and never scenario content: the episode's notion of
    "today" is the scenario's `reference_date`, and nothing here reaches a model.
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds")

# The domain the shipped scenario belongs in: facts, procedures, and the
# reasoning behind them, as opposed to the software-development store. Call
# `list_domains` on your endpoint to see what your token can reach.
DEFAULT_DOMAIN = "knowledge"

# Only the tools that open a piece of work name a domain. The ones that follow
# up on a search or a write take it from the session or operation id instead,
# and reject `domain` outright as an unexpected property.
_DOMAIN_SCOPED = frozenset({"search", "create_memory", "describe_domain"})

# `topic` is a hard filter in the knowledge domain, so the harness must use one
# fixed value everywhere: a search tagged differently from the write that
# produced the lesson matches nothing, and a run would silently look like an
# agent that never learns. It is pinned here rather than passed in, so neither a
# caller nor a model can vary it.
#
# It doubles as a fence. A trial writes invented policies for a fictional
# company into what may be a workspace someone also uses for real work; keeping
# every entry under one topic keeps this scenario's lessons out of their results.
TOPIC = "memco-lotj"
TOPIC_TAG = f'<tag type="topic" name="{TOPIC}" />'
_TOPIC_TAG = re.compile(r'<tag\b[^>]*\btype\s*=\s*"topic"', re.IGNORECASE)

# The server meters calls per minute. An episode is a burst against that meter,
# not a steady trickle: two searches, then a feedback call, then a write per
# breach, all inside a few seconds. Waiting between episodes does not help,
# because the burst is inside one.
#
# Two things keep a run under the ceiling. A floor between consecutive calls
# spreads a burst out at the point where the calls are actually made, so no
# caller has to know the limit exists. And a rate-limit refusal is retried on a
# widening wait, because it is the server asking for time rather than a failure.
MIN_CALL_SPACING = 2.0
RATE_LIMIT_BACKOFF = (5.0, 15.0, 30.0)
# Everything else keeps the one immediate retry it has always had: a transient
# transport fault is worth one more go, and waiting does not make it likelier.
OTHER_BACKOFF = (0.0,)
_RATE_LIMITED = re.compile(r"rate.?limit", re.IGNORECASE)

# Most specific first. The server's boilerplate also carries the sentence
# "include the session id '{session_id}'", where the placeholder is left
# uninterpolated, so the looser quoted form must never be tried first and any
# value still holding braces is discarded below.
_SESSION_ID_PATTERNS = (
    re.compile(r"session id to provide feedback:\s*(\S+)", re.IGNORECASE),
    re.compile(r"session id[^\n]*?['\"]([^'\"]+)['\"]", re.IGNORECASE),
)
_MEMORY_BLOCK = re.compile(r'<memory\b([^>]*)>(.*?)</memory>', re.DOTALL)
_ATTR_IDX = re.compile(r'\bidx="([^"]+)"')
# How many times the store has surfaced this entry. Measured against a live
# server it rises on retrieval, not only on a duplicate write: three consecutive
# read-only searches for the same entry returned 34, 35, 36. So it is a
# popularity count and cannot be read as the number of times knowledge was
# merged, and a reader who counts merges with it will also count their own
# searches.
_ATTR_IMPRESSIONS = re.compile(r'\bimpressions="(\d+)"')
_INSIGHT_BLOCK = re.compile(
    r'<insight\b[^>]*\btitle="([^"]*)"[^>]*\bidx="([^"]+)"[^>]*>(.*?)</insight>', re.DOTALL
)
# Writes answer with: Operation id: "create-hq18h-1" — pass it to revert_memory.
_OP_ID = re.compile(
    r"\b(?:memory|insight|op(?:eration)?)[ _-]?id[:=]?\s*['\"]?([A-Za-z0-9_.-]{6,})", re.I
)


@dataclass(frozen=True)
class Insight:
    """One lesson returned by a search."""

    idx: str  # e.g. "memory-0-insight-1"; the handle used when giving feedback
    memory_idx: str  # e.g. "memory-0"
    title: str
    content: str


@dataclass(frozen=True)
class Memory:
    idx: str
    insights: tuple[Insight, ...]
    impressions: int = 1  # times surfaced, rising on retrieval: see _ATTR_IMPRESSIONS


@dataclass(frozen=True)
class SearchResult:
    session_id: str | None
    memories: tuple[Memory, ...] = ()
    insights: tuple[Insight, ...] = ()  # flattened, in the order returned
    text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class WriteResult:
    op_id: str | None = None
    detail: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class FeedbackEntry:
    """One judgement about one retrieved lesson.

    `policy` is harness bookkeeping, not part of the feedback: it records which
    scenario policy the lesson was judged to bear on, so a later report can tell
    a breach the agent had been told about from one it had not. It is
    deliberately absent from `to_tag`; the server is told the verdict, not the
    scenario's ground truth.
    """

    idx: str
    relevant: bool
    correct: bool
    comment: str = ""
    policy: str = ""

    def to_tag(self) -> str:
        body = _escape(self.comment)
        return (
            f'<feedback idx="{self.idx}" '
            f'relevant="{str(self.relevant).lower()}" '
            f'correct="{str(self.correct).lower()}">{body}</feedback>'
        )


@dataclass(frozen=True)
class _Response:
    """One transport round trip: the tool's text, or why it could not be made."""

    text: str = ""
    error: str | None = None


@dataclass(frozen=True)
class FeedbackResult:
    detail: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class MemcoClient:
    """Calls the Memco Memory MCP over streamable HTTP.

    Every tool takes a `domain`, the store the call reads from or writes to;
    `list_domains` on the server reports the ones a token can reach. The
    harness works in one domain for a whole run, so it is a field here rather
    than a parameter on each method.

    Tags are XML strings in the shape the domain expects, for example
    `<tag type="task" name="draft-reply" />`. They are sent as plain strings;
    the server normalises them. The `topic` tag is not a caller's to set: see
    TOPIC above.
    """

    url: str
    token: str | None = None
    domain: str = DEFAULT_DOMAIN
    timeout: float = 60.0
    default_tags: tuple[str, ...] = field(default_factory=tuple)
    # Injected so the tests can watch the waiting without serving it.
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    # None until the first call: a zero would mean "a call happened at time
    # zero", which is only harmless because a real monotonic clock starts large.
    _last_call_at: float | None = field(default=None, repr=False, compare=False)

    def search(self, query: str, tags: list[str] | None = None) -> SearchResult:
        response = self._call("search", {"query": query, "tags": self._tags(tags)})
        if response.error:
            return SearchResult(session_id=None, error=response.error)
        return _parse_search(response.text)

    def create_memory(
        self,
        query: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        source: str = "agent",
    ) -> WriteResult:
        result = self._call(
            "create_memory",
            {
                "query": query,
                "title": title,
                "content": content,
                "tags": self._tags(tags),
                "source": source,
            },
        )
        if result.error:
            return WriteResult(error=result.error)
        match = _OP_ID.search(result.text)
        return WriteResult(
            op_id=match.group(1) if match else None,
            detail=result.text.strip()[:500],
        )

    def share_feedback(self, session_id: str, entries: list[FeedbackEntry]) -> FeedbackResult:
        if not entries:
            return FeedbackResult(detail="no entries")
        # The server accepts up to ten entries per call.
        result = self._call(
            "share_feedback",
            {"session_id": session_id, "feedback": [e.to_tag() for e in entries[:10]]},
        )
        if result.error:
            return FeedbackResult(error=result.error)
        return FeedbackResult(detail=result.text.strip()[:500])

    # --- transport ------------------------------------------------------------

    def _tags(self, tags: list[str] | None) -> list[str]:
        """Always exactly one topic, ours. Any other topic tag is dropped."""
        supplied = [*self.default_tags, *(tags or [])]
        return [TOPIC_TAG, *(tag for tag in supplied if not _TOPIC_TAG.search(tag))]

    def _call(self, tool: str, arguments: dict[str, Any]) -> _Response:
        """Return the tool's text output, or an error.

        The error is the one signal callers need: a memory call that fails is
        recorded and the task carries on. Success and failure are separate
        fields rather than one string, because a successful call's text is
        itself arbitrary prose and cannot be told apart from an error by
        inspection.
        """
        if tool in _DOMAIN_SCOPED:
            arguments = {"domain": self.domain, **arguments}
        last_error = ""
        attempt = 0
        while True:
            attempt += 1
            self._space()
            try:
                text = asyncio.run(self._call_async(tool, arguments))
                return _Response(text=text)
            except Exception as exc:  # noqa: BLE001 - surfaced to the result line
                last_error = _describe(exc)
            # The schedule follows the error in hand, so a rate limit that turns
            # into something else stops being waited out.
            backoff = RATE_LIMIT_BACKOFF if _RATE_LIMITED.search(last_error) else OTHER_BACKOFF
            if attempt > len(backoff):
                break
            self.sleeper(backoff[attempt - 1])
        # A refused write can still have landed: episode 12 of trial 3 had a
        # create_memory report a rate limit, and episode 13 retrieved the lesson.
        # So the error is recorded and the run carries on. What the store ends up
        # holding is the server's decision and not something a client can verify,
        # so this does not go looking: an existence check here would answer a
        # question the answer to which can still change afterwards.
        return _Response(error=f"memco {tool} failed: {last_error}")

    def _space(self) -> None:
        """Hold a floor between consecutive calls, wherever they come from.

        Smoothing here rather than at the call sites means the agent's searches,
        the feedback call and the per-breach writes all draw on one budget
        without any of them knowing about the others.
        """
        if MIN_CALL_SPACING <= 0:
            return
        if self._last_call_at is not None:
            wait = self._last_call_at + MIN_CALL_SPACING - self.clock()
            if wait > 0:
                self.sleeper(wait)
        self._last_call_at = self.clock()

    async def _call_async(self, tool: str, arguments: dict[str, Any]) -> str:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        http_client = create_mcp_http_client(headers=headers, timeout=httpx2.Timeout(self.timeout))
        async with streamable_http_client(self.url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
        if getattr(result, "is_error", False):
            raise RuntimeError(_text_of(result) or "tool reported an error")
        return _text_of(result)


def _describe(exc: BaseException) -> str:
    """Flatten an exception for the result line.

    Transport failures surface as ExceptionGroups nested a couple of levels
    deep by the anyio task groups inside the MCP client, and the outer message
    is only ever "unhandled errors in a TaskGroup". Recurse so the line in the
    JSONL names the thing that actually went wrong.

    An SDK wrapper is often the least informative frame of the lot: the
    OpenAI client raises `APIConnectionError: Connection error.` and puts the
    thing you actually need — a DNS failure, a read timeout, a reset connection
    — in `__cause__`. Following the chain is the difference between a line that
    says something went wrong and a line that says what.
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(sub) for sub in exc.exceptions)
    described = f"{type(exc).__name__}: {exc}"
    cause, seen = exc.__cause__ or exc.__context__, {id(exc)}
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        described += f" <- {type(cause).__name__}: {cause}"
        cause = cause.__cause__ or cause.__context__
    return described


def _text_of(result: Any) -> str:
    parts = [
        block.text
        for block in getattr(result, "content", []) or []
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(parts)


def _parse_search(text: str) -> SearchResult:
    """Pull the session id, memories, and insights out of a search response.

    Parsing is deliberately forgiving: if the shape changes, the raw text is
    still carried on the result and the agent still sees something useful.
    """
    session_id = None
    for pattern in _SESSION_ID_PATTERNS:
        match = pattern.search(text)
        # An uninterpolated "{session_id}" in the server's boilerplate is not an
        # id; taking it would send every share_feedback to a session that never
        # existed. Keep looking rather than carry it forward.
        if match and "{" not in match.group(1):
            session_id = match.group(1)
            break

    memories: list[Memory] = []
    insights: list[Insight] = []
    for attributes, body in _MEMORY_BLOCK.findall(text):
        idx_match = _ATTR_IDX.search(attributes)
        if not idx_match:
            continue
        memory_idx = idx_match.group(1)
        impressions = _ATTR_IMPRESSIONS.search(attributes)
        found = tuple(
            Insight(
                idx=insight_idx,
                memory_idx=memory_idx,
                title=title.strip(),
                content=_clean(content),
            )
            for title, insight_idx, content in _INSIGHT_BLOCK.findall(body)
        )
        memories.append(
            Memory(
                idx=memory_idx,
                insights=found,
                impressions=int(impressions.group(1)) if impressions else 1,
            )
        )
        insights.extend(found)
    return SearchResult(
        session_id=session_id,
        memories=tuple(memories),
        insights=tuple(insights),
        text=text,
    )


def _clean(content: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content)).strip()


def build_client() -> MemcoClient:
    """Build the client from the environment. Raises if no endpoint is set."""
    url = (os.environ.get("MEMCO_MCP_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "MEMCO_MCP_URL is not set. Point it at your Memco Memory MCP endpoint, "
            "or run with --no-memory."
        )
    return MemcoClient(
        url=url,
        token=(os.environ.get("MEMCO_MCP_TOKEN") or "").strip() or None,
        domain=(os.environ.get("MEMCO_MCP_DOMAIN") or "").strip() or DEFAULT_DOMAIN,
    )
