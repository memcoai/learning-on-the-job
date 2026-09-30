"""A thin client for Memco Shared Memory.

Exactly what the harness needs and nothing more: open a session, search memory,
write a lesson, and give feedback on what a search returned. The transport is
the official Memco Python SDK, which returns typed results, so nothing here
parses prose.

What is left is policy the SDK declines to have opinions about:

* **A session per episode.** A session is the unit of one task's work. Opening
  one and naming it on every call is what relates the searches, the feedback and
  the lessons written afterwards to the episode that produced them. A call
  naming only a domain is still valid — it just opens a throwaway session of its
  own, which is right for a read-only probe (see `report.reconcile`) and wrong
  for everything else.
* **Pacing and backoff.** The SDK retries only idempotent reads, and never a
  rate limit; both are deliberate, and both are the caller's to handle.
* **Never raising into the run.** Every method returns a result object. A memory
  call that fails is recorded on the result, written to the task's line in the
  JSONL, and the run carries on.

Calls are scoped to a domain, set once on the client. The shipped scenario uses
`knowledge`; override with MEMCO_DOMAIN.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memcoai import Memco
from memcoai.errors import (
    MemcoConfigError,
    MemcoError,
    MemcoResourceExhaustedError,
    ResourceExhaustedKind,
)
from memcoai.types import DataSource, FeedbackRating, Tag

__all__ = [
    "DEFAULT_DOMAIN",
    "MAX_FEEDBACK_ENTRIES",
    "TOPIC",
    "TOPIC_TAG",
    "DataSource",
    "FeedbackEntry",
    "FeedbackResult",
    "Insight",
    "MemcoClient",
    "Memory",
    "MemorySession",
    "SearchResult",
    "Tag",
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

# `topic` is a hard filter in the knowledge domain, so the harness must use one
# fixed value everywhere: a search tagged differently from the write that
# produced the lesson matches nothing, and a run would silently look like an
# agent that never learns. It is pinned here rather than passed in, so neither a
# caller nor a model can vary it.
#
# It doubles as a fence. A trial writes invented policies for a fictional
# company into what may be a workspace someone also uses for real work; keeping
# every entry under one topic keeps this scenario's lessons out of their results.
TOPIC = "memco-lotj-sdk"
TOPIC_TAG = Tag(type="topic", value=TOPIC)

# The server meters calls per minute. An episode is a burst against that meter,
# not a steady trickle: a session, two searches, then a feedback call, then a
# write per breach, all inside a few seconds. Waiting between episodes does not
# help, because the burst is inside one.
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
# A usage quota is the one failure that retrying cannot fix. The SDK separates
# it from a short-window rate limit on the same status code, which the old
# regex over the message text could not, so it no longer costs fifty seconds of
# backoff to find out that the plan is exhausted.
NO_BACKOFF: tuple[float, ...] = ()

# The server accepts a bounded number of ratings per call. Batches larger than
# this are split rather than truncated: three searches can retrieve more lessons
# than one call carries, and dropping the overflow would silently withhold
# feedback the reflection step actually produced.
MAX_FEEDBACK_ENTRIES = 10


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
    # How often the store has delivered this memory, this delivery included. It
    # rises on retrieval, not only on a duplicate write, so it is a popularity
    # count and cannot be read as the number of times knowledge was merged; a
    # reader who counts merges with it will also count their own searches. A
    # delivery only counts when content is actually rendered, so a memory
    # returned as a reference to an earlier search in the same session is free.
    times_served: int = 1


@dataclass(frozen=True)
class SearchResult:
    session_id: str | None
    memories: tuple[Memory, ...] = ()
    insights: tuple[Insight, ...] = ()  # flattened, in the order returned
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class WriteResult:
    op_id: str | None = None
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
    deliberately absent from `to_rating`; the server is told the verdict, not the
    scenario's ground truth.
    """

    idx: str
    relevant: bool
    correct: bool
    comment: str = ""
    policy: str = ""

    def to_rating(self) -> FeedbackRating:
        return FeedbackRating(
            idx=self.idx,
            relevant=self.relevant,
            correct=self.correct,
            comment=self.comment or None,
        )


@dataclass(frozen=True)
class FeedbackResult:
    detail: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class _Outcome:
    """One round trip: what the call returned, or why it could not be made."""

    value: Any = None
    error: str | None = None


@dataclass(frozen=True)
class MemorySession:
    """One episode's session, and the calls made inside it.

    `session_id` is None when the session could not be opened. The calls still
    go out — each one opening a throwaway session of its own, which is what the
    client did before sessions were threaded through — so a failure here costs
    the relation between an episode's calls and not the episode itself.
    """

    client: MemcoClient
    session_id: str | None = None
    error: str | None = None

    def search(self, query: str, tags: Sequence[Tag] | None = None) -> SearchResult:
        return self.client.search(query, tags=tags, session_id=self.session_id)

    def create_memory(
        self,
        query: str,
        title: str,
        content: str,
        tags: Sequence[Tag] | None = None,
        source: DataSource = DataSource.AGENT,
    ) -> WriteResult:
        return self.client.create_memory(
            query=query,
            title=title,
            content=content,
            tags=tags,
            source=source,
            session_id=self.session_id,
        )

    def share_feedback(self, session_id: str, entries: list[FeedbackEntry]) -> FeedbackResult:
        """Rate what one search returned.

        The session id is passed in rather than taken from this object because
        it belongs to the search being graded. The two are the same id whenever
        the episode's session opened, and differ when it did not.
        """
        return self.client.share_feedback(session_id, entries)


@dataclass
class MemcoClient:
    """Calls Memco Shared Memory through the SDK, paced and never raising.

    `ops` is the SDK's memory operations namespace — `Memco(...).memory`. It is
    a field rather than something built here so the tests can drive the whole
    client without a server.

    The harness works in one domain for a whole run, so `domain` is a field here
    rather than a parameter on each method. It is named on the call that opens a
    session, and on any call made without one.
    """

    ops: Any
    domain: str = DEFAULT_DOMAIN
    default_tags: tuple[Tag, ...] = field(default_factory=tuple)
    # The SDK client, held only so the channel can be closed at the end of a
    # run. None when the caller supplied `ops` directly, as the tests do.
    owner: Memco | None = field(default=None, repr=False, compare=False)
    # Injected so the tests can watch the waiting without serving it.
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    # None until the first call: a zero would mean "a call happened at time
    # zero", which is only harmless because a real monotonic clock starts large.
    _last_call_at: float | None = field(default=None, repr=False, compare=False)

    # --- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        if self.owner is not None:
            self.owner.close()

    def __enter__(self) -> MemcoClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- calls ----------------------------------------------------------------

    def open_session(self) -> MemorySession:
        """Open the session an episode's calls belong to.

        A session remembers the domain it was opened with, so nothing after this
        names one. The reverse does not hold: naming a domain does not name a
        session, it opens a new one for that single call.
        """
        outcome = self._call("start_session", self.ops.start_session, self.domain)
        if outcome.error:
            return MemorySession(client=self, error=outcome.error)
        return MemorySession(client=self, session_id=outcome.value.id)

    def search(
        self,
        query: str,
        tags: Sequence[Tag] | None = None,
        session_id: str | None = None,
    ) -> SearchResult:
        outcome = self._call(
            "search",
            self.ops.search,
            query,
            tags=self._tags(tags),
            **self._scope(session_id),
        )
        if outcome.error:
            return SearchResult(session_id=None, error=outcome.error)
        return _to_search_result(outcome.value)

    def create_memory(
        self,
        query: str,
        title: str,
        content: str,
        tags: Sequence[Tag] | None = None,
        source: DataSource = DataSource.AGENT,
        session_id: str | None = None,
    ) -> WriteResult:
        outcome = self._call(
            "create_memory",
            self.ops.create_memory,
            query=query,
            title=title,
            content=content,
            tags=self._tags(tags),
            source=source,
            **self._scope(session_id),
        )
        if outcome.error:
            return WriteResult(error=outcome.error)
        return WriteResult(op_id=outcome.value.operation_id)

    def share_feedback(self, session_id: str, entries: list[FeedbackEntry]) -> FeedbackResult:
        """Send every rating, in as many calls as the per-call bound requires."""
        if not entries:
            return FeedbackResult(detail="no entries")
        recorded = 0
        advice: list[str] = []
        for start in range(0, len(entries), MAX_FEEDBACK_ENTRIES):
            batch = entries[start : start + MAX_FEEDBACK_ENTRIES]
            outcome = self._call(
                "share_feedback",
                self.ops.share_feedback,
                session_id=session_id,
                feedback=[entry.to_rating() for entry in batch],
            )
            if outcome.error:
                return FeedbackResult(error=outcome.error)
            for recorded_entry in outcome.value.entries:
                recorded += 1
                if recorded_entry.advice:
                    advice.append(f"{recorded_entry.idx}: {recorded_entry.advice}")
        detail = f"{recorded} rating(s) recorded"
        if advice:
            detail = f"{detail}; {'; '.join(advice)}"
        return FeedbackResult(detail=detail[:500])

    # --- transport ------------------------------------------------------------

    def _scope(self, session_id: str | None) -> dict[str, str]:
        """Name the session if there is one, and the domain only if there is not.

        A session carries the domain it was opened with, so passing both is
        redundant; passing neither is refused.
        """
        return {"session_id": session_id} if session_id else {"domain": self.domain}

    def _tags(self, tags: Sequence[Tag] | None) -> list[Tag]:
        """Always exactly one topic, ours. Any other topic tag is dropped."""
        supplied = [*self.default_tags, *(tags or [])]
        return [TOPIC_TAG, *(tag for tag in supplied if not _is_topic(tag))]

    def _call(self, label: str, method: Callable[..., Any], *args: Any, **kwargs: Any) -> _Outcome:
        """Make one call, or return why it could not be made.

        The error is the one signal callers need: a memory call that fails is
        recorded and the task carries on. Success and failure are separate
        fields rather than one value, because a call can legitimately return
        nothing and that must not read as a failure.
        """
        last_error = ""
        attempt = 0
        while True:
            attempt += 1
            self._space()
            try:
                return _Outcome(value=method(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001 - surfaced to the result line
                last_error = _describe(exc)
                backoff = _backoff_for(exc)
            if attempt > len(backoff):
                break
            self.sleeper(backoff[attempt - 1])
        # A refused write can still have landed: episode 12 of trial 3 had a
        # create_memory report a rate limit, and episode 13 retrieved the lesson.
        # So the error is recorded and the run carries on. What the store ends up
        # holding is the server's decision and not something a client can verify,
        # so this does not go looking: an existence check here would answer a
        # question the answer to which can still change afterwards.
        return _Outcome(error=f"memco {label} failed: {last_error}")

    def _space(self) -> None:
        """Hold a floor between consecutive calls, wherever they come from.

        Smoothing here rather than at the call sites means the agent's searches,
        the feedback call and the per-breach writes all draw on one budget
        without any of them knowing about the others. The floor is held on the
        client rather than on a session for the same reason: the burst it exists
        to spread out happens inside a single episode.
        """
        if MIN_CALL_SPACING <= 0:
            return
        if self._last_call_at is not None:
            wait = self._last_call_at + MIN_CALL_SPACING - self.clock()
            if wait > 0:
                self.sleeper(wait)
        self._last_call_at = self.clock()


def _is_topic(tag: Tag) -> bool:
    """Whether a tag claims the topic type, in whatever spelling.

    The service lowercases the type and folds hyphens to underscores before
    matching, so the fence has to compare what the service will compare.
    """
    return tag.type.strip().lower().replace("-", "_") == "topic"


def _backoff_for(exc: BaseException) -> tuple[float, ...]:
    """How long to wait before trying this failure again, if at all.

    The schedule follows the error in hand, so a rate limit that turns into
    something else stops being waited out.
    """
    if isinstance(exc, MemcoResourceExhaustedError):
        if exc.kind is ResourceExhaustedKind.QUOTA:
            return NO_BACKOFF
        return RATE_LIMIT_BACKOFF
    return OTHER_BACKOFF


def _describe(exc: BaseException) -> str:
    """Flatten an exception for the result line.

    An SDK wrapper is often the least informative frame of the lot: the
    OpenAI client raises `APIConnectionError: Connection error.` and puts the
    thing you actually need — a DNS failure, a read timeout, a reset connection
    — in `__cause__`. Following the chain is the difference between a line that
    says something went wrong and a line that says what.

    Exception groups are flattened too. The memory client no longer produces
    them, but the runner describes episode failures with this as well, and a
    provider running its own task group still can.
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


def _to_search_result(result: Any) -> SearchResult:
    """Flatten the SDK's nested result into the shape the harness grades against.

    Insights are lifted out of their memories and kept in the order returned,
    because feedback is given per insight while retrieval is reported per
    memory. A memory the session has already returned arrives as a reference
    with no insights of its own, and so contributes nothing here — which is the
    intended effect: it is already in the agent's context.
    """
    memories: list[Memory] = []
    insights: list[Insight] = []
    for memory in result.memories:
        found = tuple(
            Insight(
                idx=insight.idx,
                memory_idx=memory.idx,
                title=insight.title.strip(),
                content=insight.content.strip(),
            )
            for insight in memory.insights
        )
        memories.append(
            Memory(idx=memory.idx, insights=found, times_served=memory.times_served)
        )
        insights.extend(found)
    return SearchResult(
        session_id=result.session_id,
        memories=tuple(memories),
        insights=tuple(insights),
    )


def build_client() -> MemcoClient:
    """Build the client from the environment. Raises if it cannot connect.

    Connecting here rather than on the first call is deliberate. The SDK checks
    the endpoint's health and reads the domain directory while constructing, so
    a wrong host or a rejected token fails before the run has spent a single
    provider token. The alternative is worse than it sounds: every memory call
    failing individually leaves the memory arm drafting with nothing to draw on,
    which is a second control arm wearing the label of the first.
    """
    domain = (os.environ.get("MEMCO_DOMAIN") or "").strip() or DEFAULT_DOMAIN
    host = (os.environ.get("MEMCO_API_HOST") or "").strip() or None
    try:
        # The SDK attaches its own handler and reports every connection at info.
        # The harness writes a formatted table to the same terminal, and a run
        # is long enough that interleaved log lines make it unreadable. Memory
        # failures have their own channel here — the task's result line and its
        # entry in the JSONL — so nothing is lost by silencing this one.
        client = Memco(host=host, timeout=60.0, log_level="none")
    except MemcoConfigError as exc:
        raise RuntimeError(f"{exc}. Alternatively, run with --no-memory.") from exc
    except MemcoError as exc:
        raise RuntimeError(
            f"could not reach Memco Shared Memory: {_describe(exc)}. "
            "Check MEMCO_API_TOKEN and MEMCO_API_HOST, or run with --no-memory."
        ) from exc
    return MemcoClient(ops=client.memory, domain=domain, owner=client)
