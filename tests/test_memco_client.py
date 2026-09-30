"""The client is policy over the SDK, so the policy is what is tested here.

The SDK returns typed results, so there is no prose to parse and nothing to
guess at. What is left is the harness's own: a session per episode, one topic on
every call, a floor between calls, and a schedule for retrying that follows the
error in hand. Fixtures are built from the SDK's own result types rather than
hand-rolled look-alikes, so a change to a shape the harness reads breaks here.

No network: the operations namespace is replaced with a recorder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import grpc
import pytest
from memcoai import types as sdk
from memcoai.errors import (
    MemcoResourceExhaustedError,
    MemcoUnavailableError,
    ResourceExhaustedKind,
)
from memcoai.operations import Session

from memco_harness.memco_client import (
    DEFAULT_DOMAIN,
    MAX_FEEDBACK_ENTRIES,
    MIN_CALL_SPACING,
    TOPIC,
    TOPIC_TAG,
    FeedbackEntry,
    MemcoClient,
    Tag,
)

INSTRUCTIONS = sdk.Instructions(content="", policy="", adding="", rating="", next="")


def sdk_insight(idx: str, title: str = "A lesson", content: str = "The body") -> sdk.Insight:
    return sdk.Insight(
        idx=idx, title=title, content=content, updated=None, times_served=1, endorsed=0, disputed=0
    )


def sdk_memory(
    idx: str, *insights: sdk.Insight, times_served: int = 1, reference: str | None = None
) -> sdk.Memory:
    return sdk.Memory(
        idx=idx,
        kind="memory",
        times_served=times_served,
        intents=(),
        insights=tuple(insights),
        reference=reference,
    )


def sdk_search(session_id: str = "session-41", *memories: sdk.Memory) -> sdk.SearchResult:
    return sdk.SearchResult(
        session_id=session_id, memories=tuple(memories), notice=None, instructions=INSTRUCTIONS
    )


def rate_limited() -> MemcoResourceExhaustedError:
    error = MemcoResourceExhaustedError(
        grpc.StatusCode.RESOURCE_EXHAUSTED, "rate limit exceeded, try again shortly"
    )
    assert error.kind is ResourceExhaustedKind.RATE_LIMIT  # the fixture, not the client
    return error


def quota_exhausted() -> MemcoResourceExhaustedError:
    error = MemcoResourceExhaustedError(
        grpc.StatusCode.RESOURCE_EXHAUSTED, "monthly quota exhausted for this plan"
    )
    assert error.kind is ResourceExhaustedKind.QUOTA
    return error


@dataclass
class FakeOps:
    """Stands in for the SDK's memory operations namespace.

    Answers every method with whatever it was told to, and records what it was
    called with, so argument assembly stays under test.
    """

    result: Any = None
    exc: Exception | None = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def _answer(self, method: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((method, arguments))
        if self.exc is not None:
            raise self.exc
        return self.result

    def start_session(self, domain: str, **kwargs: Any) -> Any:
        return self._answer("start_session", {"domain": domain, **kwargs})

    def search(self, query: str, **kwargs: Any) -> Any:
        return self._answer("search", {"query": query, **kwargs})

    def create_memory(self, **kwargs: Any) -> Any:
        return self._answer("create_memory", kwargs)

    def share_feedback(self, **kwargs: Any) -> Any:
        self.calls.append(("share_feedback", kwargs))
        if self.exc is not None:
            raise self.exc
        return sdk.FeedbackResult(
            session_id=kwargs["session_id"],
            entries=tuple(
                sdk.FeedbackEntry(
                    idx=rating.idx, relevant=rating.relevant, correct=rating.correct, advice=None
                )
                for rating in kwargs["feedback"]
            ),
            instructions=INSTRUCTIONS,
        )


class RecordingClient(MemcoClient):
    """A client whose transport records and whose waiting is served, not slept.

    The client holds a floor between calls and backs off on rate limits, and a
    suite that actually waited for those would take minutes. `waits` records
    what it would have waited for.
    """

    def __init__(self, result: Any = None, exc: Exception | None = None, **kwargs: Any) -> None:
        self.waits: list[float] = []
        self._now = 0.0
        kwargs.setdefault("sleeper", self._advance)
        kwargs.setdefault("clock", lambda: self._now)
        super().__init__(ops=FakeOps(result=result, exc=exc), **kwargs)

    def _advance(self, seconds: float) -> None:
        self.waits.append(round(seconds, 3))
        self._now += seconds

    @property
    def fake(self) -> FakeOps:
        return self.ops

    def last(self) -> dict[str, Any]:
        return self.fake.calls[-1][1]


# --- results -------------------------------------------------------------------


def test_search_flattens_insights_and_keeps_the_session():
    memory = sdk_memory("memory-0", sdk_insight("memory-0-insight-0", "Returns close in 24 days"))
    client = RecordingClient(result=sdk_search("session-41", memory))

    result = client.search("returns window")

    assert result.session_id == "session-41"
    assert [insight.idx for insight in result.insights] == ["memory-0-insight-0"]
    # Feedback is given per insight while retrieval is reported per memory, so
    # the flattened view has to carry the memory each insight came from.
    assert result.insights[0].memory_idx == "memory-0"
    assert result.memories[0].times_served == 1


def test_a_referenced_memory_contributes_no_insights():
    """A memory the session already returned arrives as a pointer, not content.

    That is the intended effect — it is already in the agent's context — and the
    flattened view must not invent an insight to stand in for it.
    """
    already_seen = sdk_memory("memory-1", reference="memory-0")
    fresh = sdk_memory("memory-2", sdk_insight("memory-2-insight-0"))
    client = RecordingClient(result=sdk_search("session-41", already_seen, fresh))

    result = client.search("anything")

    assert [insight.idx for insight in result.insights] == ["memory-2-insight-0"]
    assert [memory.idx for memory in result.memories] == ["memory-1", "memory-2"]


def test_search_success_is_not_reported_as_an_error():
    client = RecordingClient(result=sdk_search())
    assert client.search("anything").ok


def test_create_memory_carries_the_operation_id():
    client = RecordingClient(
        result=sdk.WriteResult(operation_id="create-hq18j-1", instructions=INSTRUCTIONS)
    )
    result = client.create_memory(query="q", title="t", content="c")
    assert result.ok
    assert result.op_id == "create-hq18j-1"


def test_transport_failure_surfaces_as_an_error():
    client = RecordingClient(exc=MemcoUnavailableError(grpc.StatusCode.UNAVAILABLE, "no route"))
    result = client.search("anything")
    assert not result.ok
    # display.py tells a memory outage from an episode failure by this prefix.
    assert result.error.startswith("memco search failed:")
    assert "no route" in result.error


def test_feedback_reports_what_the_store_recorded():
    client = RecordingClient()
    result = client.share_feedback(
        "session-41", [FeedbackEntry(idx="memory-0-insight-0", relevant=True, correct=True)]
    )
    assert result.ok
    assert "1 rating(s) recorded" in result.detail


def test_feedback_is_split_rather_than_truncated():
    """Three searches can retrieve more lessons than one call carries.

    Truncating would silently withhold verdicts the reflection step actually
    produced, and the run would report them as sent.
    """
    client = RecordingClient()
    entries = [
        FeedbackEntry(idx=f"memory-{i}-insight-0", relevant=True, correct=True)
        for i in range(MAX_FEEDBACK_ENTRIES + 2)
    ]

    result = client.share_feedback("session-41", entries)

    calls = [call for call in client.fake.calls if call[0] == "share_feedback"]
    assert [len(call[1]["feedback"]) for call in calls] == [MAX_FEEDBACK_ENTRIES, 2]
    assert f"{MAX_FEEDBACK_ENTRIES + 2} rating(s) recorded" in result.detail


# --- retrying ------------------------------------------------------------------


def test_a_rate_limit_is_waited_out_on_a_widening_delay():
    client = RecordingClient(exc=rate_limited())
    client.search("anything")
    # The floor before the first call is not a wait; the rest are the schedule.
    assert client.waits == [5.0, 15.0, 30.0]


def test_a_persistent_rate_limit_gives_up_after_three_retries():
    client = RecordingClient(exc=rate_limited())
    result = client.search("anything")
    assert not result.ok
    assert len([call for call in client.fake.calls if call[0] == "search"]) == 4


def test_a_quota_is_not_waited_out():
    """A rate limit is the server asking for time. A quota is not.

    They arrive on the same status code, and the SDK is what separates them, so
    the plan being exhausted no longer costs fifty seconds to discover.
    """
    client = RecordingClient(exc=quota_exhausted())
    result = client.search("anything")
    assert not result.ok
    assert client.waits == []
    assert len([call for call in client.fake.calls if call[0] == "search"]) == 1


def test_other_errors_keep_their_single_immediate_retry():
    """Immediate, but still paced.

    The backoff for a transient fault is zero because waiting does not make the
    retry likelier — but the floor between calls is about the server's meter,
    not about this failure, so the retry still waits its turn. The rate-limit
    schedule above shows no such entry because every step of it already exceeds
    the floor.
    """
    client = RecordingClient(exc=MemcoUnavailableError(grpc.StatusCode.UNAVAILABLE, "reset"))
    client.search("anything")
    assert client.waits == [0.0, MIN_CALL_SPACING]
    assert len([call for call in client.fake.calls if call[0] == "search"]) == 2


# --- pacing --------------------------------------------------------------------


def test_consecutive_calls_are_spaced_so_a_burst_is_smoothed():
    client = RecordingClient(result=sdk_search())
    client.search("one")
    client.search("two")
    assert client.waits == [2.0]


def test_time_already_spent_counts_towards_the_spacing():
    client = RecordingClient(result=sdk_search())
    client.search("one")
    client._advance(1.5)  # a model call, say
    client.waits.clear()
    client.search("two")
    assert client.waits == [0.5]


# --- scoping -------------------------------------------------------------------


def test_a_call_without_a_session_names_the_domain():
    client = RecordingClient(result=sdk_search())
    client.search("anything")
    assert client.last()["domain"] == DEFAULT_DOMAIN
    assert "session_id" not in client.last()


def test_a_session_supersedes_the_domain():
    """A session carries the domain it was opened with.

    Sending both is redundant, and sending the domain instead of the session is
    the mistake that quietly turns one task's work into unrelated calls.
    """
    client = RecordingClient(result=sdk_search())
    client.search("anything", session_id="session-41")
    assert client.last()["session_id"] == "session-41"
    assert "domain" not in client.last()


def test_domain_defaults_to_the_knowledge_store():
    assert RecordingClient().domain == DEFAULT_DOMAIN


def test_an_open_session_is_named_by_every_call_made_through_it():
    client = RecordingClient(
        result=Session(None, "session-41", INSTRUCTIONS, tool_catalog=None)
    )
    session = client.open_session()
    assert session.session_id == "session-41"
    assert client.fake.calls[0] == ("start_session", {"domain": DEFAULT_DOMAIN})

    client.fake.result = sdk_search()
    session.search("anything")
    assert client.last()["session_id"] == "session-41"

    client.fake.result = sdk.WriteResult(operation_id="create-1", instructions=INSTRUCTIONS)
    session.create_memory(query="q", title="t", content="c")
    assert client.last()["session_id"] == "session-41"


def test_a_session_that_could_not_be_opened_degrades_to_domain_scoped_calls():
    """A failure here costs the relation between an episode's calls, not the run.

    Each call then opens a throwaway session of its own, which is what the
    client did before sessions were threaded through, so the episode still
    drafts, still grades and still writes.
    """
    client = RecordingClient(exc=MemcoUnavailableError(grpc.StatusCode.UNAVAILABLE, "no route"))
    session = client.open_session()
    assert session.session_id is None
    assert session.error.startswith("memco start_session failed:")

    client.fake.exc = None
    client.fake.result = sdk_search()
    assert session.search("anything").ok
    assert client.last()["domain"] == DEFAULT_DOMAIN
    assert "session_id" not in client.last()


# --- the topic fence -----------------------------------------------------------


@pytest.mark.parametrize("method", ["search", "create_memory"])
def test_search_and_write_carry_the_same_single_topic(method: str):
    """A read tagged differently from the write would match nothing.

    Which would look exactly like an agent that never learns, so it is pinned in
    one place and asserted for both directions.
    """
    client = RecordingClient()
    if method == "search":
        client.fake.result = sdk_search()
        client.search("anything")
    else:
        client.fake.result = sdk.WriteResult(operation_id="create-1", instructions=INSTRUCTIONS)
        client.create_memory(query="q", title="t", content="c")
    assert [tag for tag in client.last()["tags"] if tag.type == "topic"] == [TOPIC_TAG]


def test_a_caller_supplied_topic_is_discarded():
    client = RecordingClient(result=sdk_search())
    client.search("anything", tags=[Tag(type="topic", value="someone-elses")])
    assert [tag for tag in client.last()["tags"] if tag.type == "topic"] == [TOPIC_TAG]


def test_a_topic_is_recognised_however_it_is_spelled():
    """The service lowercases the type and folds hyphens, so the fence must too."""
    client = RecordingClient(result=sdk_search())
    client.search("anything", tags=[Tag(type="TOPIC", value="someone-elses")])
    assert [tag.value for tag in client.last()["tags"] if tag.type.lower() == "topic"] == [TOPIC]


def test_default_tags_cannot_smuggle_a_topic_in_either():
    client = RecordingClient(
        result=sdk_search(), default_tags=(Tag(type="topic", value="someone-elses"),)
    )
    client.search("anything")
    assert [tag for tag in client.last()["tags"] if tag.type == "topic"] == [TOPIC_TAG]


def test_non_topic_tags_are_passed_through():
    client = RecordingClient(result=sdk_search())
    client.search("anything", tags=[Tag(type="task", value="draft-reply")])
    assert Tag(type="task", value="draft-reply") in client.last()["tags"]


def test_the_scenario_tags_declare_no_topic_of_their_own():
    from memco_harness import agent, reflection

    for tags in (agent.MEMORY_TAGS, reflection.MEMORY_TAGS):
        assert not [tag for tag in tags if tag.type == "topic"]


def test_topic_is_namespaced_to_this_harness():
    """A trial writes invented policies into what may be a real workspace.

    The `-sdk` suffix additionally fences this branch's runs off from those the
    MCP client made under `memco-lotj`, so the two are not measured together.
    """
    assert TOPIC == "memco-lotj-sdk"
