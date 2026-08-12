"""The simulated human reviewer.

This is the component you would replace with your own QA process, so its
interface is kept clean: give it the task, the draft, and the records, and it
gives back a breach list and a corrected draft. If your team already reviews
agent output, that is the seam where this harness meets your real workflow.

The reviewer holds the hidden policies. The metric is the breach list and never
edit distance: a correction that only changes tone carries no penalty.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .providers import Message, Provider
from .scenario import Account, Order, Policy, Task, Variant

__all__ = ["Breach", "Review", "Reviewer"]

SYSTEM_PROMPT = """\
You are an experienced order-desk supervisor at Fenmoor Supplies. A junior \
assistant has drafted a reply to a customer. You know the desk's policies; the \
assistant does not. Check the draft against the policies you are given, correct \
it, and record what was wrong.

Today's date is {today}. Every window, deadline, and age is counted from that \
date; do not use any other notion of today.

Judge only the policies listed. Do not invent policies, and do not flag \
anything outside the list. A draft that satisfies every stated obligation has \
no breaches, even if you would have phrased it differently: style is not a \
breach.

The obligations given under each policy are the ground truth for this email: \
they already state what the correct reply must do. Judge against them.

The breach list and the corrected draft are two views of the same set of \
problems, and they must agree. Every change you make to the draft for a policy \
reason has an entry in the breach list, and every entry in the breach list \
shows up as a change in the corrected draft. Before you answer, read your \
corrected draft back against the original and check that nothing you fixed is \
missing from the list. A correction made silently is a breach that never got \
counted.

Your notes and your corrected draft have different readers, and different \
rules follow from that. The corrected draft goes to the customer, so anything \
the desk keeps to itself stays out of it. The notes never leave the desk: they \
are read by the colleague working out what the assistant should have known, \
and by nobody else. So a note says what made the policy apply here, not only \
what the draft got wrong.

Give that condition in terms of what the assistant could have seen for itself \
by looking the records up: the account's credit standing or its tier, how long \
it has traded, the order's status or what has been paid on it, the category of \
the goods, how long ago something was delivered. "The reply gave delivery \
timing, which this account's credit standing does not allow" is a usable note. \
"The reply gave delivery timing" is not, because it reads as a rule about \
every reply that mentions timing, and that is what it will become.

Where a policy forbids telling the customer something, that restraint binds \
the corrected draft alone. Write the reason plainly in the note.

Reply with JSON and nothing else, in this shape:

{"breaches": [{"policy": "<policy id>", "note": "<what was wrong, and the \
condition that made the policy apply>"}], "corrected_draft": "<the reply as it \
should be sent>"}

The corrected draft must satisfy every obligation, read as a reply the customer \
would actually receive, and keep whatever the assistant got right."""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _policy_key(value: str) -> str:
    """A policy id in the form it can be matched, whatever decoration it carries.

    The scope filter used to compare the reviewer's id to the task's exactly,
    which sounds safe and was not. Each policy is put in front of the reviewer
    under a heading of `[its-id] Its title`, so the model copied the bracketed
    form into its answer; the filter then found no such policy in scope, dropped
    the breach, and recorded a reply that had breached nothing. Eight control
    episodes of a hundred came back spotless that way, and a spotless control
    episode is not a thing this scenario produces.

    The lesson generalises past the brackets: an id is content the model
    retypes, and it will retype it quoted, cased differently, or with a stray
    space. None of those is a different policy, and none of them should be able
    to delete a breach. What must still be dropped is a policy that is genuinely
    not in scope, which is what the filter is for.
    """
    return value.strip().strip("[]").strip("\"'").strip().casefold()


@dataclass(frozen=True)
class Breach:
    policy: str
    note: str


@dataclass(frozen=True)
class Review:
    breaches: tuple[Breach, ...] = ()
    corrected_draft: str = ""
    failed: bool = False
    dropped: tuple[str, ...] = ()
    error: str = ""
    raw: str = ""  # kept only on failure, so a bad review can be read back

    @property
    def breach_count(self) -> int:
        return len(self.breaches)


@dataclass
class Reviewer:
    provider: Provider
    policies: dict[str, Policy]
    reference_date: str

    @property
    def system_prompt(self) -> str:
        # Literal replacement, not str.format: the prompt shows the reviewer a
        # JSON template, and its braces are not placeholders.
        return SYSTEM_PROMPT.replace("{today}", self.reference_date)

    def review(
        self,
        task: Task,
        variant: Variant,
        draft: str,
        account: Account,
        order: Order | None,
    ) -> Review:
        prompt = self._prompt(task, variant, draft, account, order)
        messages = [Message(role="user", text=prompt)]

        last = ""
        for attempt in (1, 2):
            completion = self.provider.complete(self.system_prompt, messages, tools=None)
            last = completion.text
            parsed = _parse(completion.text)
            if parsed is not None:
                return self._to_review(task, parsed)
            if attempt == 1:
                messages = [
                    Message(role="user", text=prompt),
                    Message(role="assistant", text=completion.text),
                    Message(
                        role="user",
                        text=(
                            "That was not valid JSON. Reply again with the JSON object "
                            "only: keys 'breaches' and 'corrected_draft', nothing else."
                        ),
                    ),
                ]
        # Guessing here would put noise straight into the metric.
        return Review(
            failed=True,
            error="reviewer output was not parseable JSON",
            raw=last,
        )

    def _to_review(self, task: Task, parsed: dict) -> Review:
        allowed = {_policy_key(policy_id): policy_id for policy_id in task.applicable_policies}
        breaches: list[Breach] = []
        dropped: list[str] = []
        seen: set[str] = set()
        for entry in parsed.get("breaches") or []:
            if not isinstance(entry, dict):
                continue
            named = str(entry.get("policy", "")).strip()
            policy_id = allowed.get(_policy_key(named))
            if policy_id is None:
                if named:
                    dropped.append(named)
                continue
            if policy_id in seen:
                continue
            seen.add(policy_id)
            breaches.append(Breach(policy=policy_id, note=str(entry.get("note", "")).strip()))
        return Review(
            breaches=tuple(breaches),
            corrected_draft=str(parsed.get("corrected_draft", "")).strip(),
            dropped=tuple(dropped),
        )

    def _prompt(
        self,
        task: Task,
        variant: Variant,
        draft: str,
        account: Account,
        order: Order | None,
    ) -> str:
        policy_block = "\n\n".join(
            self._policy_text(policy_id, task) for policy_id in task.applicable_policies
        )
        order_block = json.dumps(order.as_record(), indent=2) if order else "none referenced"
        return (
            "POLICIES IN SCOPE\n"
            f"{policy_block}\n\n"
            "ACCOUNT RECORD\n"
            f"{json.dumps(account.as_record(), indent=2)}\n\n"
            "ORDER RECORD\n"
            f"{order_block}\n\n"
            "CUSTOMER EMAIL\n"
            f"Subject: {variant.subject}\n{variant.body}\n\n"
            "ASSISTANT'S DRAFT\n"
            f"{draft or '(the assistant produced no draft)'}"
        )

    def _policy_text(self, policy_id: str, task: Task) -> str:
        policy = self.policies[policy_id]
        obligations = [o.obligation for o in task.expected_obligations if o.policy == policy_id]
        lines = [
            f"[{policy.id}] {policy.title}",
            f"Policy: {policy.statement}",
            f"How to check: {policy.check_hint}",
        ]
        for obligation in obligations:
            lines.append(f"On this email: {obligation}")
        return "\n".join(lines)


def _parse(text: str) -> dict | None:
    """Pull a JSON object out of the reviewer's reply, tolerating fences."""
    if not text:
        return None
    candidates = [match.strip() for match in _FENCE.findall(text)]
    candidates.append(text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and "corrected_draft" in parsed:
            return parsed
    return None
