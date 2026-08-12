"""The adapter's job is normalisation, so this checks the shapes it produces.

The Anthropic API validates tool schemas strictly and one malformed schema
rejects the whole request, so the harness's tool definitions are round-tripped
through both adapters here against mocked SDK clients. Nothing hits the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memco_harness.agent import LOOKUP_ACCOUNT, LOOKUP_ORDER, MEMORY_SEARCH
from memco_harness.providers import (
    DEFAULT_OPENAI_BASE_URL,
    AnthropicProvider,
    Message,
    OpenAIProvider,
    ProviderError,
    ToolCall,
    Usage,
    _to_anthropic_messages,
    _to_openai_messages,
    build_provider,
    model_for,
    split_spec,
)

TOOLS = [MEMORY_SEARCH, LOOKUP_ACCOUNT, LOOKUP_ORDER]

CONVERSATION = [
    Message(role="user", text="Where has my order got to?"),
    Message(
        role="assistant",
        text="Let me check.",
        tool_calls=[ToolCall(id="call-1", name="lookup_order", arguments={"order_id": "ord-1"})],
    ),
    Message(role="tool_result", text='{"status": "dispatched"}', tool_call_id="call-1"),
]


class FakeAnthropic:
    """Captures the request and answers with one text block and one tool call."""

    def __init__(self) -> None:
        self.captured: dict = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=120, output_tokens=34),
            content=[
                SimpleNamespace(type="text", text="Checking now."),
                SimpleNamespace(
                    type="tool_use",
                    id="call-9",
                    name="lookup_account",
                    input={"account_id": "acc-101"},
                ),
            ],
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.captured: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured = kwargs
        message = SimpleNamespace(
            content="Checking now.",
            tool_calls=[
                SimpleNamespace(
                    id="call-9",
                    function=SimpleNamespace(
                        name="lookup_account",
                        arguments=json.dumps({"account_id": "acc-101"}),
                    ),
                )
            ],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=34),
        )


def anthropic_provider() -> tuple[AnthropicProvider, FakeAnthropic]:
    provider = object.__new__(AnthropicProvider)
    provider.model = "test-model"
    provider.spec = "anthropic:test-model"
    provider.usage = Usage()
    client = FakeAnthropic()
    provider._client = client
    return provider, client


def openai_provider() -> tuple[OpenAIProvider, FakeOpenAI]:
    provider = object.__new__(OpenAIProvider)
    provider.model = "test-model"
    provider.spec = "openai:test-model"
    provider.usage = Usage()
    provider.adaptations = set()
    client = FakeOpenAI()
    provider._client = client
    return provider, client


def test_tool_schemas_round_trip_through_the_anthropic_adapter():
    provider, client = anthropic_provider()
    completion = provider.complete("system", CONVERSATION, tools=TOOLS)

    sent = {tool["name"]: tool for tool in client.captured["tools"]}
    assert set(sent) == {"memory_search", "lookup_account", "lookup_order"}
    for tool in TOOLS:
        schema = sent[tool.name]["input_schema"]
        assert schema == tool.parameters
        assert schema["type"] == "object"
        assert set(schema["required"]) <= set(schema["properties"])
        assert sent[tool.name]["description"] == tool.description

    assert completion.text == "Checking now."
    assert completion.tool_calls[0].name == "lookup_account"
    assert completion.tool_calls[0].arguments == {"account_id": "acc-101"}


def test_tool_schemas_round_trip_through_the_openai_adapter():
    provider, client = openai_provider()
    completion = provider.complete("system", CONVERSATION, tools=TOOLS)

    sent = {tool["function"]["name"]: tool["function"] for tool in client.captured["tools"]}
    assert set(sent) == {"memory_search", "lookup_account", "lookup_order"}
    for tool in TOOLS:
        assert sent[tool.name]["parameters"] == tool.parameters
        assert sent[tool.name]["description"] == tool.description

    assert completion.text == "Checking now."
    assert completion.tool_calls[0].arguments == {"account_id": "acc-101"}


def test_anthropic_tool_results_are_grouped_into_one_user_turn():
    messages = [
        Message(role="user", text="hello"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="a", name="lookup_account", arguments={}),
                ToolCall(id="b", name="lookup_order", arguments={}),
            ],
        ),
        Message(role="tool_result", text="{}", tool_call_id="a"),
        Message(role="tool_result", text="{}", tool_call_id="b"),
    ]
    turns = _to_anthropic_messages(messages)
    assert [turn["role"] for turn in turns] == ["user", "assistant", "user"]
    assert len(turns[-1]["content"]) == 2, "parallel tool results belong in a single turn"


def test_openai_tool_results_are_separate_tool_turns():
    turns = _to_openai_messages(CONVERSATION)
    assert [turn["role"] for turn in turns] == ["user", "assistant", "tool"]
    assert turns[1]["tool_calls"][0]["function"]["name"] == "lookup_order"
    assert turns[2]["tool_call_id"] == "call-1"


def test_the_system_prompt_leads_the_openai_request():
    provider, client = openai_provider()
    provider.complete("you are the order desk", CONVERSATION, tools=TOOLS)
    assert client.captured["messages"][0] == {
        "role": "system",
        "content": "you are the order desk",
    }


def test_a_refusal_is_reported_rather_than_read_as_a_draft():
    provider, client = anthropic_provider()

    def refuse(**kwargs):
        client.captured = kwargs
        return SimpleNamespace(
            stop_reason="refusal",
            content=[],
            usage=SimpleNamespace(input_tokens=12, output_tokens=0),
        )

    client.messages.create = refuse
    completion = provider.complete("system", CONVERSATION, tools=TOOLS)
    assert completion.refused
    assert completion.text == ""


def test_a_truncated_reply_is_flagged_rather_than_read_as_finished():
    provider, client = anthropic_provider()

    def truncate(**kwargs):
        client.captured = kwargs
        return SimpleNamespace(
            stop_reason="max_tokens",
            content=[SimpleNamespace(type="text", text="Half a sen")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=8000),
        )

    client.messages.create = truncate
    assert provider.complete("system", CONVERSATION).truncated


def test_both_adapters_accumulate_token_usage():
    for provider, _ in (anthropic_provider(), openai_provider()):
        provider.complete("system", CONVERSATION, tools=TOOLS)
        provider.complete("system", CONVERSATION, tools=TOOLS)
        assert provider.usage == Usage(input_tokens=240, output_tokens=68, calls=2)


# --- construction from provider:model -----------------------------------------


def test_a_spec_splits_into_provider_and_model():
    assert split_spec("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")
    assert split_spec(" OpenAI : gpt-5.6-luna ") == ("openai", "gpt-5.6-luna")


@pytest.mark.parametrize("spec", ["claude-sonnet-5", "anthropic:", ":model", "mistral:large"])
def test_a_malformed_spec_is_rejected_with_the_string_in_the_message(spec):
    with pytest.raises(ProviderError, match="anthropic|provider"):
        split_spec(spec)


def test_roles_may_name_different_providers_in_one_run(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "anthropic:claude-sonnet-5")
    monkeypatch.setenv("REVIEWER_MODEL", "openai:gpt-5.6-luna")
    monkeypatch.delenv("REFLECTION_MODEL", raising=False)
    assert model_for("agent") == "anthropic:claude-sonnet-5"
    assert model_for("reviewer") == "openai:gpt-5.6-luna"
    assert model_for("reflection") == "anthropic:claude-haiku-4-5"


def test_an_empty_openai_base_url_falls_back_to_the_openai_endpoint(monkeypatch):
    """`.env.example` ships OPENAI_BASE_URL blank.

    Left to itself the SDK reads that variable and honours the empty string, so
    every request goes out with no scheme and fails as a connection error. The
    adapter must pass an explicit URL rather than None.
    """
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = build_provider("openai:gpt-5.6-luna")
    assert str(provider._client.base_url).rstrip("/") == DEFAULT_OPENAI_BASE_URL.rstrip("/")


def test_an_explicit_openai_base_url_is_honoured(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.mistral.ai/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = build_provider("openai:mistral-large-latest")
    assert "mistral" in str(provider._client.base_url)
    assert provider.spec == "openai:mistral-large-latest"


# --- compatibility adaptations ------------------------------------------------


def rejecting_client(client: FakeOpenAI, message: str, reject_while):
    """Wrap the fake so it 400s with `message` while `reject_while(request)` holds."""
    inner = client._create

    def create(**kwargs):
        if reject_while(kwargs):
            raise RuntimeError(f"Error code: 400 - {message}")
        return inner(**kwargs)

    client.chat.completions.create = create


def test_an_endpoint_that_wants_max_tokens_gets_it():
    provider, client = openai_provider()
    rejecting_client(
        client,
        "Unsupported parameter: 'max_completion_tokens'",
        lambda request: "max_completion_tokens" in request,
    )
    provider.complete("system", CONVERSATION, tools=TOOLS, max_tokens=1234)
    assert client.captured["max_tokens"] == 1234
    assert "max_completion_tokens" not in client.captured
    assert provider.adaptations == {"max_tokens"}


def test_a_reasoning_model_that_refuses_tools_has_reasoning_switched_off():
    """Some reasoning models reject function tools on chat/completions.

    The adaptation is recorded because it changes how the model works, and a run
    that quietly disabled reasoning would report a harder scenario than it ran.
    """
    provider, client = openai_provider()
    rejecting_client(
        client,
        "Function tools with reasoning_effort are not supported ... "
        "or set reasoning_effort to 'none'.",
        lambda request: "reasoning_effort" not in request,
    )
    provider.complete("system", CONVERSATION, tools=TOOLS)
    assert client.captured["reasoning_effort"] == "none"
    assert provider.adaptations == {"reasoning_effort_none"}


def test_a_learned_adaptation_is_applied_to_every_later_request():
    provider, client = openai_provider()
    rejecting_client(
        client,
        "set reasoning_effort to 'none'",
        lambda request: "reasoning_effort" not in request,
    )
    provider.complete("system", CONVERSATION, tools=TOOLS)
    provider.complete("system", CONVERSATION, tools=TOOLS)
    # Two rejected calls would mean the adapter re-learned it; one means it stuck.
    assert provider.usage.calls == 2
    assert client.captured["reasoning_effort"] == "none"


def test_an_unrecognised_error_is_raised_rather_than_adapted_around():
    provider, client = openai_provider()
    rejecting_client(client, "the model is on fire", lambda request: True)
    with pytest.raises(RuntimeError, match="on fire"):
        provider.complete("system", CONVERSATION, tools=TOOLS)
    assert provider.adaptations == set()


def test_no_sampling_parameters_are_sent():
    """Current Anthropic models reject temperature, top_p, and top_k."""
    provider, client = anthropic_provider()
    provider.complete("system", CONVERSATION, tools=TOOLS)
    assert not {"temperature", "top_p", "top_k"} & set(client.captured)
