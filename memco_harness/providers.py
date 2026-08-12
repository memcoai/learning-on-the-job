"""LLM adapter: one interface, two implementations.

The rest of the harness never sees a provider-specific structure. It builds
`Message` and `ToolDef` values, calls `complete()`, and reads back a
`Completion`. Normalising here is the adapter's whole job; the official SDKs
handle transport, retries, and versioning.

Each role names its model as `provider:model`, so one run can hold both adapters
at once: an Anthropic agent under an OpenAI-compatible reviewer, or the reverse.
Providers are built per role and only for the providers a role actually names,
so a run that never uses OpenAI never needs an OpenAI key.

A note on the Anthropic path: tool schemas are validated strictly and one
malformed schema rejects the whole request, so the harness keeps its tool
definitions minimal, and `tests/test_providers.py` checks that they round-trip
through both adapters.

Sampling parameters are deliberately not sent. Current Anthropic models reject
`temperature`, `top_p`, and `top_k`, and the harness does not need them: the
prompts do the steering.

Every completion carries its token usage, and each provider accumulates a
running total, which is how a run reports what it cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

__all__ = [
    "Completion",
    "Message",
    "Provider",
    "ProviderError",
    "ToolCall",
    "ToolDef",
    "Usage",
    "build_provider",
    "model_for",
    "split_spec",
]

Role = Literal["user", "assistant", "tool_result"]

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Each role's default, as a full provider:model spec. Kept in step with
# `.env.example`, which is what a reader will copy.
DEFAULT_MODELS = {
    "agent": "anthropic:claude-sonnet-5",
    "reviewer": "anthropic:claude-haiku-4-5",
    "reflection": "anthropic:claude-haiku-4-5",
}


class ProviderError(RuntimeError):
    """Raised when a provider cannot be built or a completion cannot be read."""


@dataclass(frozen=True)
class Usage:
    """Tokens spent. Summed across a run to report what it cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
        )

    def as_record(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
        }


@dataclass(frozen=True)
class ToolDef:
    """A tool offered to the model. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One turn in the internal conversation shape.

    `tool_result` messages carry the output of a single tool call; consecutive
    ones are merged into a single provider turn where the provider requires it.
    """

    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    refused: bool = False
    truncated: bool = False  # stopped at max_tokens with more to say
    usage: Usage = Usage()


class Provider(Protocol):
    """What the harness needs from a model."""

    model: str  # the bare model id the SDK is called with
    spec: str  # the full provider:model string, recorded in every result line
    usage: Usage  # running total across this provider's calls

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Completion: ...


# --- Anthropic ----------------------------------------------------------------


class AnthropicProvider:
    """The `anthropic` SDK."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError("the 'anthropic' package is not installed") from exc
        self.model = model
        self.spec = f"anthropic:{model}"
        self.usage = Usage()
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": _to_anthropic_messages(messages),
        }
        if tools:
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        response = self._client.messages.create(**request)
        usage = _anthropic_usage(response)
        self.usage = self.usage + usage

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            return Completion(text="", refused=True, usage=usage)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return Completion(
            text="".join(text_parts),
            tool_calls=tuple(calls),
            truncated=stop_reason == "max_tokens",
            usage=usage,
        )


def _anthropic_usage(response: Any) -> Usage:
    raw = getattr(response, "usage", None)
    return Usage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        calls=1,
    )


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            out.append({"role": "user", "content": message.text})
        elif message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.text:
                blocks.append({"type": "text", "text": message.text})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.text,
            }
            # Results for tool calls made in the same assistant turn belong in
            # one user turn; splitting them teaches the model to stop calling
            # tools in parallel.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


# --- OpenAI-compatible --------------------------------------------------------


class OpenAIProvider:
    """The `openai` SDK, pointed at any OpenAI-compatible endpoint."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError("the 'openai' package is not installed") from exc
        self.model = model
        self.spec = f"openai:{model}"
        self.usage = Usage()
        # Which compatibility adaptations this endpoint turned out to need. They
        # change what the model does, so a run records them rather than papering
        # over them: see `_ADAPTATIONS`.
        self.adaptations: set[str] = set()
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        # Always pass a base URL. Left to itself the SDK reads OPENAI_BASE_URL
        # from the environment, and it honours an empty string rather than
        # ignoring one: `.env.example` ships the variable blank, so a user who
        # copies it and fills in only their key would otherwise get requests
        # with no scheme, surfacing as "APIConnectionError: Connection error."
        self._client = OpenAI(base_url=base_url or DEFAULT_OPENAI_BASE_URL, **kwargs)

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *_to_openai_messages(messages)],
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        response = self._create(request)
        usage = _openai_usage(response)
        self.usage = self.usage + usage
        choice = response.choices[0]
        calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_loads_or_empty(call.function.arguments),
            )
            for call in (choice.message.tool_calls or [])
        )
        return Completion(
            text=choice.message.content or "",
            tool_calls=calls,
            truncated=getattr(choice, "finish_reason", None) == "length",
            usage=usage,
        )

    def _create(self, request: dict[str, Any]) -> Any:
        """Send the request, adapting once per known compatibility difference.

        An adaptation is learned once and then applied to every later request,
        so an endpoint that needs one pays a single rejected call, not one per
        episode.
        """
        for name in self.adaptations:
            _apply_adaptation(request, name)
        for _ in range(len(_ADAPTATIONS) + 1):
            try:
                return self._client.chat.completions.create(**request)
            except Exception as exc:  # noqa: BLE001 - re-raised unless we recognise it
                if not self._adapt(request, str(exc)):
                    raise
        raise ProviderError(f"{self.spec}: request rejected after every known adaptation")

    def _adapt(self, request: dict[str, Any], message: str) -> bool:
        """Apply the first untried adaptation the error asks for. False if none fits."""
        for needle, name in _ADAPTATIONS:
            if needle not in message or name in self.adaptations:
                continue
            if not _apply_adaptation(request, name):
                continue
            self.adaptations.add(name)
            return True
        return False


# "OpenAI-compatible" is a family, not a spec, and the members disagree about a
# few request fields. Each entry is (what the error says, what to change), tried
# once each against a rejected request. Anything else is re-raised.
_ADAPTATIONS: tuple[tuple[str, str], ...] = (
    # Older servers (Mistral among them) know `max_tokens` but not the newer name.
    ("max_completion_tokens", "max_tokens"),
    # Some reasoning models decline function tools on /v1/chat/completions
    # unless reasoning is switched off. This one is not cosmetic: it changes how
    # the model works the problem, so the run records that it happened.
    ("reasoning_effort", "reasoning_effort_none"),
)


def _apply_adaptation(request: dict[str, Any], name: str) -> bool:
    """Rewrite `request` for one known difference. False if it does not apply."""
    if name == "max_tokens":
        if "max_completion_tokens" not in request:
            return False
        request["max_tokens"] = request.pop("max_completion_tokens")
        return True
    if name == "reasoning_effort_none":
        request["reasoning_effort"] = "none"
        return True
    return False


def _openai_usage(response: Any) -> Usage:
    raw = getattr(response, "usage", None)
    return Usage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        calls=1,
    )


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            out.append({"role": "user", "content": message.text})
        elif message.role == "assistant":
            turn: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                turn["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            out.append(turn)
        else:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.text,
                }
            )
    return out


def _loads_or_empty(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# --- Construction -------------------------------------------------------------


def split_spec(spec: str) -> tuple[str, str]:
    """Split `provider:model` into its two halves, with a clear error if it is not."""
    kind, separator, model = spec.strip().partition(":")
    kind, model = kind.strip().lower(), model.strip()
    if not separator or not kind or not model:
        raise ProviderError(
            f"{spec!r} is not a provider:model string; "
            "expected something like 'anthropic:claude-sonnet-5'"
        )
    if kind not in {"anthropic", "openai"}:
        raise ProviderError(
            f"unknown provider {kind!r} in {spec!r}: expected 'anthropic' or 'openai'"
        )
    return kind, model


def build_provider(spec: str) -> Provider:
    """Build the provider named by a `provider:model` spec, reading the environment.

    Keys are read only for the provider actually named, so a run whose roles all
    sit on one provider never needs the other's credentials.
    """
    kind, model = split_spec(spec)
    if kind == "anthropic":
        return AnthropicProvider(model=model, api_key=os.environ.get("ANTHROPIC_API_KEY") or None)
    return OpenAIProvider(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
        base_url=(os.environ.get("OPENAI_BASE_URL") or "").strip() or None,
    )


def model_for(role: str) -> str:
    """The configured `provider:model` for a role: agent, reviewer, or reflection."""
    if role not in DEFAULT_MODELS:
        raise ProviderError(f"unknown role {role!r}: expected one of {', '.join(DEFAULT_MODELS)}")
    return (os.environ.get(f"{role.upper()}_MODEL") or "").strip() or DEFAULT_MODELS[role]
