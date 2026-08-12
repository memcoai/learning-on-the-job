"""The client parses prose, so the shapes the server actually sends matter.

Every case here is a verbatim excerpt of a live Spark Memory 0.3.0 response.
No network: the transport is replaced with a canned reply.
"""

from __future__ import annotations

from memco_harness.memco_client import (
    DEFAULT_DOMAIN,
    TOPIC,
    TOPIC_TAG,
    FeedbackEntry,
    MemcoClient,
    _parse_search,
)

# The tail of a real search response, kept in the shape a server once sent: the
# boilerplate carried an uninterpolated "{session_id}" ahead of the real id, and
# picking that up sent every share_feedback to a session that never existed. The
# server interpolates it now; this guards the parse order regardless.
SEARCH_TEXT = """The following information are relevant to the users intent.

## Memories

<memory idx="memory-0" impressions="3">
### Insights
<insight title="Returns close 24 days after delivery" idx="memory-0-insight-0">
A return of stocked goods is accepted within 24 days of the delivery date.
</insight>
</memory>

## Feedback Instructions

Always make sure to include the session id '{session_id}' provided in the recommendation.

## Session ID

Use this session id to provide feedback: session-id-41
"""

WRITE_TEXT = (
    "Memory saved successfully. Your knowledge will be processed and made available. "
    'Operation id: "create-hq18j-1" — pass it to revert_memory to undo this write.'
)


class RecordingClient(MemcoClient):
    """Stands in for the transport only, so argument assembly is still under test.

    Time is injected rather than served: the client waits between calls and
    backs off on rate limits, and a test suite that actually slept for those
    would take minutes. `waits` records what it would have waited for.
    """

    def __init__(self, text: str = "", exc: Exception | None = None, **kwargs):
        self.waits: list[float] = []
        self._now = 0.0
        kwargs.setdefault("sleeper", self._advance)
        kwargs.setdefault("clock", lambda: self._now)
        super().__init__(url="https://example.invalid/mcp", **kwargs)
        self._text = text
        self._exc = exc
        self.calls: list[tuple[str, dict]] = []

    def _advance(self, seconds: float) -> None:
        self.waits.append(round(seconds, 3))
        self._now += seconds

    async def _call_async(self, tool, arguments):
        self.calls.append((tool, arguments))
        if self._exc is not None:
            raise self._exc
        return self._text


def test_placeholder_session_id_is_not_mistaken_for_the_real_one():
    result = _parse_search(SEARCH_TEXT)
    assert result.session_id == "session-id-41"


def test_search_extracts_memories_and_insights():
    result = _parse_search(SEARCH_TEXT)
    assert [m.idx for m in result.memories] == ["memory-0"]
    assert [i.idx for i in result.insights] == ["memory-0-insight-0"]
    assert "24 days" in result.insights[0].content


def test_search_success_is_not_reported_as_an_error():
    client = RecordingClient(SEARCH_TEXT)
    result = client.search("how long is the returns window?")
    assert result.ok
    assert result.error is None
    assert result.session_id == "session-id-41"


def test_create_memory_parses_the_operation_id():
    client = RecordingClient(WRITE_TEXT)
    result = client.create_memory(query="q", title="t", content="c")
    assert result.ok
    assert result.op_id == "create-hq18j-1"


def test_transport_failure_surfaces_as_an_error():
    client = RecordingClient(exc=ConnectionError("nope"))
    result = client.search("anything")
    assert not result.ok
    assert "ConnectionError: nope" in (result.error or "")


# --- staying under the server's per-minute ceiling -----------------------------


class RateLimited(RuntimeError):
    """What the server sends when a burst outruns its meter."""

    def __init__(self, fail_times: int) -> None:
        super().__init__("MCPError: rate limit exceeded")
        self.fail_times = fail_times


def flaky_client(fail_times: int) -> RecordingClient:
    """Fails with a rate limit `fail_times` times, then succeeds."""
    client = RecordingClient(SEARCH_TEXT)
    remaining = {"n": fail_times}

    async def maybe(tool, arguments):
        client.calls.append((tool, arguments))
        if remaining["n"] > 0:
            remaining["n"] -= 1
            raise RuntimeError("MCPError: rate limit exceeded")
        return SEARCH_TEXT

    client._call_async = maybe
    return client


def test_a_rate_limit_is_waited_out_on_a_widening_delay():
    client = flaky_client(fail_times=2)
    result = client.search("q")
    assert result.ok, "the server asked for time, not for the call to be abandoned"
    assert len(client.calls) == 3
    # The 2s floor before each attempt, with the backoff waits interleaved.
    assert 5.0 in client.waits and 15.0 in client.waits
    assert 30.0 not in client.waits, "it succeeded before the last step"


def test_a_persistent_rate_limit_gives_up_after_three_retries():
    client = flaky_client(fail_times=99)
    result = client.search("q")
    assert not result.ok
    assert "rate limit" in (result.error or "")
    assert len(client.calls) == 4, "the first attempt plus three retries"
    assert [w for w in client.waits if w in (5.0, 15.0, 30.0)] == [5.0, 15.0, 30.0]


def test_other_errors_keep_their_single_immediate_retry():
    """Waiting does not make a transport fault likelier to clear."""
    client = RecordingClient(exc=ConnectionError("nope"))
    client.search("q")
    assert len(client.calls) == 2
    assert not [w for w in client.waits if w >= 5.0], "no backoff for a non-rate-limit error"


def test_consecutive_calls_are_spaced_so_a_burst_is_smoothed():
    """An episode makes its memory calls in a burst, not a trickle."""
    client = RecordingClient(SEARCH_TEXT)
    client.search("first")
    client.search("second")
    client.search("third")
    assert client.waits == [2.0, 2.0], "two gaps for three calls, and none before the first"


def test_time_already_spent_counts_towards_the_spacing():
    """A slow call has already provided the gap; waiting again would be wasted."""
    client = RecordingClient(SEARCH_TEXT)
    client.search("first")
    client._now += 10.0  # the next call happens well after the floor
    client.search("second")
    assert client.waits == [], "nothing to wait for"


def test_only_domain_scoped_tools_are_sent_a_domain():
    """share_feedback derives its domain from the session and rejects the argument."""
    client = RecordingClient(SEARCH_TEXT, domain="knowledge")
    client.search("q")
    client.create_memory(query="q", title="t", content="c")
    entry = FeedbackEntry(idx="memory-0", relevant=True, correct=True)
    client.share_feedback("session-id-41", [entry])
    sent = {tool: args for tool, args in client.calls}
    assert sent["search"]["domain"] == "knowledge"
    assert sent["create_memory"]["domain"] == "knowledge"
    assert "domain" not in sent["share_feedback"]


def test_domain_defaults_to_the_knowledge_store():
    assert MemcoClient(url="https://example.invalid/mcp").domain == DEFAULT_DOMAIN == "knowledge"


# --- the topic tag is a hard filter, so it has to be identical on every call ---


def topics(tags: list[str]) -> list[str]:
    return [t for t in tags if 'type="topic"' in t]


def test_search_and_write_carry_the_same_single_topic():
    """Differing topics between read and write would return nothing, silently."""
    client = RecordingClient(SEARCH_TEXT)
    client.search("q")
    client.create_memory(query="q", title="t", content="c")
    sent = {tool: args for tool, args in client.calls}
    assert topics(sent["search"]["tags"]) == [TOPIC_TAG]
    assert topics(sent["create_memory"]["tags"]) == [TOPIC_TAG]


def test_a_caller_supplied_topic_is_discarded():
    client = RecordingClient(SEARCH_TEXT)
    client.search("q", tags=['<tag type="topic" name="order-desk" />'])
    assert topics(client.calls[0][1]["tags"]) == [TOPIC_TAG]


def test_default_tags_cannot_smuggle_a_topic_in_either():
    client = RecordingClient(SEARCH_TEXT, default_tags=('<tag type="topic" name="sneaky" />',))
    client.search("q")
    assert topics(client.calls[0][1]["tags"]) == [TOPIC_TAG]


def test_non_topic_tags_are_passed_through():
    client = RecordingClient(SEARCH_TEXT)
    client.search("q", tags=['<tag type="task" name="draft-reply" />'])
    assert client.calls[0][1]["tags"] == [TOPIC_TAG, '<tag type="task" name="draft-reply" />']


def test_the_scenario_tags_declare_no_topic_of_their_own():
    from memco_harness import agent, reflection

    assert topics(agent.MEMORY_TAGS) == []
    assert topics(reflection.MEMORY_TAGS) == []


def test_topic_is_namespaced_to_this_harness():
    assert TOPIC == "memco-lotj"
    assert TOPIC_TAG == '<tag type="topic" name="memco-lotj" />'
