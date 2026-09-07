"""The drafting agent.

It sees the customer email and nothing else: no policies, no expected
obligations, no reviewer internals. What it can do is search Memco memory for
lessons from earlier tasks and look up the account and order behind the request.
Everything it knows about how this desk works, beyond the records, has to have
come from a lesson somebody's correction produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .memco_client import Insight, MemorySession, Tag, stamp
from .providers import Completion, Message, Provider, ToolCall, ToolDef
from .scenario import Scenario, Task, Variant
from .stems import CRAFT_STEM, SITUATION_STEM

__all__ = ["Draft", "SearchRecord", "draft_reply", "system_prompt"]

MAX_TOOL_ITERATIONS = 6

# Two searches are what the prompt asks for; a third is tolerated because some
# tasks genuinely have two sides to look up. Past that the agent is re-fetching
# what it already has under new phrasings, which fills its context with repeats
# and buys nothing.
MAX_SEARCHES = 3

# The prompt is assembled from blocks rather than written out twice, because the
# blind arm needs the same instructions minus everything about memory, and two
# hand-maintained copies would drift apart on the first edit to either.
_OPENING = """\
You are the order-desk assistant at Fenmoor Supplies, a B2B distributor. A \
customer has written in. Draft the reply that will be sent back to them.

Today's date is {today}. Work out any deadline, age, or window from that date; \
do not use any other notion of today."""

_SEARCHES = """\
Before you draft, search memory twice, using these two questions. Keep the \
wording as it is and fill in only the part in angle brackets:

1. {situation_stem}
   Fill it with what this customer is asking for and the circumstances you can \
see: the state of the account and of the order, from your lookups.

2. {craft_stem}
   Fill it with the kind of reply you are about to write — for instance "when \
responding to a return request", or simply "when writing any reply".

The wording of those two questions is not a matter of taste. What this desk \
knows was recorded under the same two forms, and a question asked another way \
does not reach it.

Much of how this desk works was never written down. Memory is where it has \
been recorded, and a lesson that applies takes precedence over your own \
instincts. The two searches find different things, so make both of them."""

# "Also" only makes sense after the searches, so the blind arm opens plainly.
_LOOKUPS = """\
{lead} the lookup tools before relying on any fact about the account or the \
order. Do not assume a value you have not looked up."""

_DRAFTING = """\
When you draft:
- Be concise and businesslike. A short reply is a good reply.
- Answer what the customer actually asked.
- Commit only to what the records{lessons} support."""

_LESSON_CHECKS = """\
- A lesson applies only when its stated conditions match the situation in front \
of you. Before relying on a retrieved lesson, check its conditions against the \
customer's request and against the account and order records you have looked \
up. Where the conditions differ, the lesson does not apply to this reply, \
however similar it sounds, and the records in front of you take precedence over \
it.
- Before you finalise, read the draft back against each lesson you retrieved, \
and revise it wherever it does not follow a lesson that applies."""

_CLOSING = """\
Reply with the body of the email only: no subject line, no notes about your \
reasoning, no square-bracket placeholders."""


def system_prompt(reference_date: str, memory: bool = True) -> str:
    """The drafting prompt, with the scenario's clock and the two stems in it.

    The stems are substituted rather than typed out here so that the question
    this agent asks and the question a lesson is filed under cannot drift apart:
    they are one string, used twice.

    `memory=False` builds the blind arm's prompt, which says nothing about
    memory, lessons or the two searches. An arm without the `memory_search` tool
    can act on none of it, and instructions it cannot follow are not neutral:
    they take up its attention and ask it to weigh knowledge it does not have.
    The two arms then differ in what they are told by exactly as much as they
    differ in what they can do, which is what makes the pair a measurement of
    memory rather than of two prompts.
    """
    blocks = [_OPENING]
    if memory:
        blocks.append(_SEARCHES)
    blocks.append(_LOOKUPS.replace("{lead}", "Also use" if memory else "Use"))
    drafting = _DRAFTING.replace("{lessons}", " and the lessons" if memory else "")
    if memory:
        drafting = f"{drafting}\n{_LESSON_CHECKS}"
    blocks.extend([drafting, _CLOSING])
    return (
        "\n\n".join(blocks)
        .replace("{today}", reference_date)
        .replace("{situation_stem}", SITUATION_STEM)
        .replace("{craft_stem}", CRAFT_STEM)
    )

# The description carries no example and names nothing from the scenario: an
# example here would be steering, and the agent would echo its shape instead of
# deriving a query from the work in front of it.
MEMORY_SEARCH = ToolDef(
    name="memory_search",
    description=(
        "Search the desk's memory for lessons recorded from earlier work. "
        "Ask it, in a sentence, the question you actually want answered."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question you want lessons about, as a short sentence.",
            }
        },
        "required": ["query"],
    },
)

LOOKUP_ACCOUNT = ToolDef(
    name="lookup_account",
    description=(
        "Look up an account: name, tier, credit status, how many years they have "
        "been a customer, whether they are trading or closing, and any corporate "
        "group they belong to."
    ),
    parameters={
        "type": "object",
        "properties": {
            "account_id": {"type": "string", "description": "For example acc-117."}
        },
        "required": ["account_id"],
    },
)

LOOKUP_ORDER = ToolDef(
    name="lookup_order",
    description=(
        "Look up an order: status, payment method, goods and delivery values, "
        "total including VAT, delivery or estimate dates, carrier, credit check, "
        "payments received, and the line items with their categories and quantities."
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "For example ord-3308."}
        },
        "required": ["order_id"],
    },
)

# No topic here: the client pins that to memco_client.TOPIC for every call, and
# drops any other, because the knowledge domain treats topic as a hard filter.
MEMORY_TAGS = [
    Tag(type="task", value="draft-reply"),
]


@dataclass(frozen=True)
class SearchRecord:
    """One memory search the agent made while drafting.

    The session id is what a later `share_feedback` call is scoped to, and the
    insight `idx` values are the handles it uses, so the two travel together.
    `at` is when the search went out, which is what makes it possible to say how
    long a lesson took to become searchable after it was written.
    """

    query: str
    session_id: str | None
    insights: tuple[Insight, ...]
    error: str | None = None
    at: str = field(default_factory=stamp)


@dataclass(frozen=True)
class Draft:
    text: str
    searches: tuple[SearchRecord, ...] = ()
    tool_calls: int = 0
    truncated: bool = False  # the model ran out of output tokens mid-reply

    @property
    def retrieved(self) -> tuple[Insight, ...]:
        return tuple(insight for search in self.searches for insight in search.insights)

    @property
    def memory_errors(self) -> tuple[str, ...]:
        return tuple(s.error for s in self.searches if s.error)


def draft_reply(
    task: Task,
    variant: Variant,
    scenario: Scenario,
    provider: Provider,
    memory: MemorySession | None,
) -> Draft:
    """Draft a reply to one customer email.

    With `memory` set to None the agent has no `memory_search` tool, which is
    the control arm: same task, same records, no lessons.
    """
    tools = [LOOKUP_ACCOUNT, LOOKUP_ORDER]
    if memory is not None:
        tools = [MEMORY_SEARCH, *tools]

    system = system_prompt(scenario.reference_date, memory=memory is not None)
    messages = [Message(role="user", text=_email(task, variant))]
    searches: list[SearchRecord] = []
    # Which lessons this episode has already put in front of the agent. Inside
    # one session the server does this itself: a memory an earlier search
    # returned comes back as a reference carrying no insights, so the overlap
    # never reaches here. This is the fallback for the episode whose session
    # could not be opened, where each search runs in a session of its own and
    # the overlap does return.
    shown: set[tuple[str, str]] = set()
    call_count = 0

    for _ in range(MAX_TOOL_ITERATIONS):
        completion = provider.complete(system, messages, tools=tools)
        if not completion.tool_calls:
            return Draft(
                text=completion.text.strip(),
                searches=tuple(searches),
                tool_calls=call_count,
                truncated=completion.truncated,
            )
        messages.append(_assistant_turn(completion))
        for call in completion.tool_calls:
            call_count += 1
            messages.append(
                Message(
                    role="tool_result",
                    text=_run_tool(call, scenario, memory, searches, shown),
                    tool_call_id=call.id,
                )
            )

    # The cap is a backstop, not a normal path: ask once more without tools so
    # the task still produces a draft to review.
    completion = provider.complete(system, messages, tools=None)
    return Draft(
        text=completion.text.strip(),
        searches=tuple(searches),
        tool_calls=call_count,
        truncated=completion.truncated,
    )


def _email(task: Task, variant: Variant) -> str:
    account = task.account_id
    order = f"\nOrder referenced: {task.order_id}" if task.order_id else ""
    return (
        f"Account: {account}{order}\n"
        f"Subject: {variant.subject}\n\n"
        f"{variant.body}"
    )


def _assistant_turn(completion: Completion) -> Message:
    return Message(
        role="assistant",
        text=completion.text,
        tool_calls=list(completion.tool_calls),
    )


def _run_tool(
    call: ToolCall,
    scenario: Scenario,
    memory: MemorySession | None,
    searches: list[SearchRecord],
    shown: set[tuple[str, str]],
) -> str:
    if call.name == "lookup_account":
        return _dump(scenario.lookup_account(str(call.arguments.get("account_id", ""))))
    if call.name == "lookup_order":
        return _dump(scenario.lookup_order(str(call.arguments.get("order_id", ""))))
    if call.name == "memory_search":
        if memory is None:
            return "memory is not available on this run"
        if len(searches) >= MAX_SEARCHES:
            # Not an error, and not recorded as a search: nothing went out. Say
            # so plainly so the agent stops asking and starts writing.
            return (
                f"You have already searched {MAX_SEARCHES} times. Draft the reply "
                "with the lessons you have."
            )
        query = str(call.arguments.get("query", "")).strip()
        result = memory.search(query, tags=MEMORY_TAGS)
        searches.append(
            SearchRecord(
                query=query,
                session_id=result.session_id,
                insights=result.insights,
                error=result.error,
            )
        )
        if not result.ok:
            return f"memory search failed: {result.error}"
        return _render_lessons(result.insights, shown)
    return f"no tool named {call.name!r}"


def _render_lessons(insights: tuple[Insight, ...], shown: set[tuple[str, str]]) -> str:
    """Show the agent the lessons, and only the lessons, and each of them once.

    The server wraps search results in prose of its own, some of it addressed to
    the caller in the imperative ("REQUIRED", "you MUST call create_memory").
    Building the tool result from parsed insights rather than forwarding the
    response text keeps that prose out of the agent's context entirely: what a
    memory server says about itself is not an instruction to the order desk.

    A lesson already shown this episode is left out of the second search's
    results. The searches overlap by design and the repeats are pure noise in
    the agent's context. Within one session the server withholds them first, so
    this mostly has nothing left to do; it still covers the episode whose
    session failed to open.

    One consequence is worth naming, because it changed a measurement. An
    insight both searches return is now graded once for the episode rather than
    once per search: there is one session, it is returned once, and the second
    search reports it as a reference. That is the more honest count — two
    ratings for one judgement inflated whatever the store makes of them — but it
    is a different number from the one earlier runs recorded.
    """
    fresh = []
    for insight in insights:
        key = (insight.title, insight.content)
        if key in shown:
            continue
        shown.add(key)
        fresh.append(insight)
    if not fresh:
        return (
            "No lessons beyond the ones already listed above."
            if insights
            else "No lessons in memory match that yet."
        )
    lines = ["Lessons from memory:"]
    for insight in fresh:
        lines.append(f"- {insight.title}: {insight.content}")
    return "\n".join(lines)


def _dump(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True)
