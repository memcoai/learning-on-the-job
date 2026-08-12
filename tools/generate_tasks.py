#!/usr/bin/env python
"""Authoring-time task generator.

End users never need this to run the harness. It ships because the point of the
data/code split is that you can build a scenario for your own domain, and this
is what we used to expand the policy set into task instances.

    uv run python tools/generate_tasks.py --count 60 \
        --ledger scenario/exposure-ledger.yaml --out staging/

Without a ledger it draws policy combinations at random, which is enough to see
the shape of the thing. With one, it draws against a quota: the ledger says how
many tasks each policy should end up in and what has to be true of the account
and the order for that policy to bite, and the generator spends the quota down
across batches. What has already been spent is counted from the task files
themselves, so the ledger is never rewritten and the tasks stay the record of
what exists.

`scenario/exposure-ledger.yaml` is the ledger that produced the shipped 200
tasks, and its header documents every key. Run the command above with
`--dry-run` against it to see the quotas already spent down to nothing; write
your own by copying it and starting the counts over. The `subject` vocabulary
in it is this scenario's: those keys are read by `matches()` below, which knows
the shipped account and order fields by name, so a different domain means
adding predicates there as well as writing a new ledger.

The two house-style policies are attached rather than drawn, and their
obligations are written here rather than asked for: they say the same thing on
every task, and a model paraphrasing them each time is a source of drift for no
gain.

Variants are generated from a fixed facts block, which is how the
variants-vary-wording-only rule is enforced: the facts are decided before any
wording exists, and every generated task is put through the scenario validator
before it is written out.

Output goes to a staging directory, never straight into `scenario/tasks/`.
Every task in this repository was read by a person before it was committed, and
yours should be too: the generator gets the shape right, but only a human can
tell whether an email sounds like something a customer would actually send.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memco_harness.providers import Message, build_provider, model_for  # noqa: E402
from memco_harness.scenario import (  # noqa: E402
    Account,
    Order,
    Policy,
    ScenarioError,
    build_task,
    load_accounts,
    load_orders,
    load_policies,
)

SCENARIO = REPO_ROOT / "scenario"

# The draw takes one to three; a fourth is admitted only when the situation
# itself puts it in play, which is what the implication sweep decides. Plus the
# two house-style policies, that is the validator's ceiling of six per task.
MAX_SUBSTANTIVE = 4

SYSTEM_PROMPT = """\
You write test cases for the order desk at Fenmoor Supplies, a fictional B2B
distributor. Each case is one inbound customer email plus the notes a reviewer
will grade the desk's reply against. You are given an account record, an order
record, and one or more of the desk's internal policies. Write a case that puts
those policies in play.

Which direction the email goes, because everything else depends on it: the
customer is writing TO Fenmoor. You are never writing Fenmoor's reply. The
customer is asking, chasing, complaining or requesting, and does not know what
the answer will be. If what you have written contains a decision, a price, a
fee, an offer, a refusal, an apology on Fenmoor's behalf, or a phrase such as
"we can" or "I'm afraid we can't", then you have written the reply, and the case
is spoilt. Read each variant back and ask who sent it.

What the customer may know, and how they say it:

- They have never seen the policy book. They do not use its vocabulary: no
  "tier", no "restocking fee", no "credit check", no "window", no "lead time".
- They also do not name the remedy in the desk's terms. Asking for "a goodwill
  credit", "a price match", "a partial shipment" tells the reader which policy
  is in play, which is the whole thing being tested. They ask for "something off
  for the trouble", "a bit of money back", "whether you can match a price I have
  been quoted", "whether you can send what you have now and the rest later".
  The test to apply to every sentence: would somebody who has never seen the
  policy book put it that way?
- They cannot quote a figure they have no way of knowing. What is on their own
  order or invoice is fair game; the desk's thresholds, limits, windows and
  percentages are not, and neither is any number that would give one away.
- Everything they refer to comes from the records you were given. Do not invent
  an order, a date, or an amount.
- Dates the way a person writes them, "27 August" or "27/08", never the form the
  record stores them in.
- British spelling.

How the emails should read. Real inboxes are not uniform, and a batch of
identically terse, identically polite messages tells a reader where the test is
before they have read a word of it:

- Length varies. Some are two lines; some run to a paragraph or two with
  context. Across a batch aim anywhere from about 15 to about 120 words, and
  do not make them all short.
- Most have a sender's name and some greeting or sign-off, in whatever style
  suits them: "Hi", "Morning", "Thanks in advance", "Regards, Dave", or nothing
  at all. Invent ordinary British names; they are not in the records.
- Register varies: neutral, chatty, businesslike, mildly fed up. Somebody who is
  annoyed is still writing to a supplier they deal with every week.
- Let some of them mention something incidental that is true to the records but
  carries no policy: which site it is going to, that the last delivery went
  smoothly, that they will be away Friday. Not every email, but enough that the
  load-bearing sentences are not the only ones there.

You will be told how many variants to write, and in what register, and whether
this customer bothers with a greeting. Follow the instruction you are given
rather than your own preference: the variety is a property of the whole set, and
each of these emails is written without sight of the others. Every variant
states the same facts: every
identifier, amount and date appears in all of them identically, so a paraphrase
can never move the ground truth. What may differ between variants is the
wording, the register, the greeting and sign-off, and the incidental detail.
Each variant gets its own subject line.

Be careful where those two rules meet. Incidental colour is free to change, but
it must not bring a number with it: no figure, date, quantity or identifier may
appear in one variant and not another, or read differently between them. If you
give one variant a detail the others lack, give it no numbers. Before you
finish, read the variants side by side and check that the same set of figures
and identifiers appears in each.

The facts block: the slot values the email is built from, in the customer's
terms. What they are asking for, and any identifier, amount or date they state.
Not the account's tier, tenure or standing, and not a count of days: those are
in the records the reviewer already has, and repeating them here says the case
was written from the answer.

The obligations, which the reviewer reads and the assistant never sees:

- One per policy, in the order the policies were listed, using the policy ids
  given to you verbatim.
- Each states two things: the condition in this situation that makes the policy
  apply, and the behaviour it therefore requires. Name the account or order fact
  the condition rests on. "acc-231 is on credit hold, so the reply is the
  accounts referral alone and gives no timing of any kind" has both halves;
  "handle the credit hold correctly" has neither.
- Some of the policies you were given are there because the situation triggers
  them, not because the email is mainly about them. Write their obligations with
  the same care, and make sure the set reads as one consistent account of what
  the reply must do. Where two policies bear on the same decision, say which one
  governs: goods that are never returnable are refused on the category, whatever
  the window would otherwise have allowed.
- Where a policy has something that is applied but never said to the customer,
  the obligation says both parts: what is applied, and what is never mentioned.
- Where a policy leaves the desk a discretion or a range, the obligation gives
  the limit and what has to be said about it, never a particular choice inside
  it. "Up to 7% may be applied, and any discount given states the percentage and
  the order value" is gradeable; "apply 7%" makes one permitted reply the only
  one.

Reply with JSON and nothing else:

{"facts": {"requested_action": "...", "...": "..."},
 "variants": [{"subject": "...", "body": "..."}],
 "expected_obligations": [{"policy": "<policy id>", "obligation": "<the \
condition, and what it requires>"}]}"""


# --- the ledger ---------------------------------------------------------------


@dataclass
class Rule:
    """One policy's line in the ledger."""

    policy_id: str
    quota: int
    solo: int = 0
    alone: bool = False
    subject: dict[str, Any] = field(default_factory=dict)
    max_with: dict[str, int] = field(default_factory=dict)
    needs_companion: tuple[str, ...] = ()
    implied_by: tuple[str, ...] = ()
    implied_when: dict[str, Any] = field(default_factory=dict)
    obligation_hint: str = ""


@dataclass
class Ledger:
    """How many tasks each policy should end up in, and what it takes to get there."""

    rules: dict[str, Rule]
    sizes: dict[int, int]
    style_always: str = ""
    style_with_order: str = ""
    account_level_share: float = 0.0
    spent: Counter[str] = field(default_factory=Counter)
    solo_spent: Counter[str] = field(default_factory=Counter)
    pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    sizes_spent: Counter[int] = field(default_factory=Counter)

    @classmethod
    def load(cls, path: Path, policies: dict[str, Policy]) -> Ledger:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = data.get("policies") or {}
        rules: dict[str, Rule] = {}
        for policy_id, entry in entries.items():
            if policy_id not in policies:
                raise SystemExit(f"{path}: unknown policy {policy_id!r}")
            entry = entry or {}
            rules[policy_id] = Rule(
                policy_id=policy_id,
                quota=int(entry.get("quota", 0)),
                solo=int(entry.get("solo", 0)),
                alone=bool(entry.get("alone", False)),
                subject=entry.get("subject") or {},
                max_with={k: int(v) for k, v in (entry.get("max_with") or {}).items()},
                needs_companion=tuple(entry.get("needs_companion") or ()),
                implied_by=tuple(entry.get("implied_by") or ()),
                implied_when=entry.get("implied_when") or entry.get("subject") or {},
                obligation_hint=" ".join((entry.get("obligation_hint") or "").split()),
            )
        style = data.get("style_policies") or {}
        return cls(
            rules=rules,
            sizes={int(k): int(v) for k, v in (data.get("combination_sizes") or {}).items()},
            style_always=style.get("always", ""),
            style_with_order=style.get("with_order", ""),
            account_level_share=float(data.get("account_level_share") or 0.0),
        )

    @classmethod
    def unconstrained(cls, policies: dict[str, Policy]) -> Ledger:
        """No ledger given: every policy equally likely, nothing runs out."""
        rules = {p: Rule(policy_id=p, quota=10**6) for p in policies}
        return cls(rules=rules, sizes={})

    # --- what has already been spent ------------------------------------------

    def count_existing(self, directories: list[Path]) -> int:
        """Spend the ledger against the tasks that already exist."""
        seen = 0
        for directory in directories:
            for path in sorted(directory.rglob("*.yaml")):
                entry = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(entry, dict):
                    continue
                chosen = [p for p in (entry.get("applicable_policies") or [])
                          if p in self.rules]
                if not chosen:
                    continue
                self.record(chosen)
                seen += 1
        return seen

    def record(self, chosen: list[str]) -> None:
        for policy_id in chosen:
            self.spent[policy_id] += 1
        if len(chosen) == 1:
            self.solo_spent[chosen[0]] += 1
        for first in chosen:
            for second in chosen:
                if first < second:
                    self.pairs[(first, second)] += 1
        self.sizes_spent[len(chosen)] += 1

    def remaining(self, policy_id: str) -> int:
        return max(0, self.rules[policy_id].quota - self.spent[policy_id])

    def exhausted(self) -> bool:
        """Every quota spent, with tasks still to write."""
        return not any(self.remaining(policy_id) for policy_id in self.rules)

    def weight(self, policy_id: str) -> int:
        """How much this policy wants drawing next.

        While quota remains that is simply what is left of it. Once the table is
        spent and the library is still short, the weight falls back to the
        original k: the plan allows padding within tiers, and padding by the
        table's own shape keeps the relative frequencies it was designed around,
        where padding by whatever happened to be left last would not.
        """
        return self.remaining(policy_id) or (
            self.rules[policy_id].quota if self.exhausted() else 0
        )

    # --- drawing --------------------------------------------------------------

    def draw_size(self, rng: random.Random) -> int:
        if not self.sizes:
            return rng.randint(1, 3)
        left = {size: total - self.sizes_spent[size] for size, total in self.sizes.items()}
        left = {size: n for size, n in left.items() if n > 0}
        if not left:
            left = {size: 1 for size in self.sizes}
        return rng.choices(list(left), weights=list(left.values()))[0]

    def solo_owed(self) -> list[str]:
        return [r.policy_id for r in self.rules.values()
                if r.solo - self.solo_spent[r.policy_id] > 0 and self.remaining(r.policy_id)]

    def alone_needs(self) -> dict[str, int]:
        """Policies whose whole quota has to come out of the single-policy tasks.

        Drawing them at their share of the entire ledger would starve them: they
        compete only for the single-policy slots, which are a minority of the
        library, while the weighting is against everything.
        """
        return {r.policy_id: self.remaining(r.policy_id)
                for r in self.rules.values() if r.alone and self.remaining(r.policy_id)}

    def single_slots_left(self) -> int:
        return max(0, self.sizes.get(1, 0) - self.sizes_spent[1]) if self.sizes else 0

    def may_join(self, chosen: list[str], candidate: str) -> bool:
        if candidate in chosen or not self.weight(candidate):
            return False
        rule = self.rules[candidate]
        if rule.alone or any(self.rules[other].alone for other in chosen):
            return False
        for other in chosen:
            limit = rule.max_with.get(other, self.rules[other].max_with.get(candidate))
            if limit is not None:
                key = (candidate, other) if candidate < other else (other, candidate)
                if self.pairs[key] >= limit:
                    return False
        return True

    def satisfied(self, chosen: list[str]) -> bool:
        """Companion requirements are checked once the combination is complete."""
        return all(
            not self.rules[policy_id].needs_companion
            or any(other in self.rules[policy_id].needs_companion for other in chosen)
            for policy_id in chosen
        )


# --- who a task can be about --------------------------------------------------


@dataclass(frozen=True)
class Subject:
    account: Account
    order: Order | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.account.id, self.order.id if self.order else "")


def subjects(accounts: dict[str, Account], orders: dict[str, Order]) -> list[Subject]:
    """Every account, alone and paired with each of its orders."""
    pool = [Subject(account, None) for account in accounts.values()]
    pool += [Subject(accounts[order.account_id], order) for order in orders.values()]
    return pool


def matches(subject: Subject, selector: dict[str, Any], reference_date: dt.date) -> bool:
    """Whether a policy could bite on this account and order."""
    order = subject.order
    wants_order = selector.get("order")
    if wants_order == "required" and order is None:
        return False
    if wants_order == "none" and order is not None:
        return False

    account = subject.account
    for key, field_name in (("account_tier", "tier"),
                            ("account_credit_status", "credit_status"),
                            ("account_status", "status")):
        allowed = selector.get(key)
        if allowed and getattr(account, field_name) not in allowed:
            return False
    if "account_in_group" in selector and bool(account.group) != selector["account_in_group"]:
        return False

    order_keys = [key for key in selector if key.startswith("order_")]
    if not order_keys:
        return True
    if order is None:
        # An order-shaped condition cannot hold of a task with no order.
        return False

    if (allowed := selector.get("order_status")) and order.status not in allowed:
        return False
    categories = {line.category for line in order.lines}
    if (allowed := selector.get("order_category")) and not categories & set(allowed):
        return False
    if (barred := selector.get("order_category_not")) and categories & set(barred):
        return False
    for key, value in (("order_total_min", order.total_inc_vat),
                       ("order_goods_min", order.goods_value_ex_vat)):
        if key in selector and value < selector[key]:
            return False
    for key, value in (("order_total_max", order.total_inc_vat),
                       ("order_goods_max", order.goods_value_ex_vat)):
        if key in selector and value > selector[key]:
            return False
    if selector.get("order_has_payments") and not order.payments:
        return False
    if selector.get("order_has_backorder") and not any(
        line.backorder_working_days for line in order.lines
    ):
        return False
    for key, inside in (("order_delivered_within", True), ("order_delivered_beyond", False)):
        if key in selector:
            if not order.delivered_date:
                return False
            days = (reference_date - dt.date.fromisoformat(order.delivered_date)).days
            if inside and days > selector[key]:
                return False
            if not inside and days <= selector[key]:
                return False
    return True


def unclaimed(pool: list[Subject], ledger: Ledger, reference_date: dt.date) -> list[Subject]:
    """The subjects no precedence-taking policy has a claim on.

    A policy marked `alone` takes precedence over whatever else the situation
    touches, so on its subjects there is no second policy to grade: an on-hold
    account asking about a discount is owed the referral, not the discount rule.
    Drawing anything else there would write an obligation that contradicts the
    policy that actually governs, which is the one error class that corrupts
    the measurement rather than merely reading oddly.
    """
    claims = [rule.subject for rule in ledger.rules.values() if rule.alone and rule.subject]
    if not claims:
        return pool
    return [s for s in pool
            if not any(matches(s, claim, reference_date) for claim in claims)]


def implied(subject: Subject, chosen: list[str], ledger: Ledger,
            reference_date: dt.date) -> list[str]:
    """The policies this situation puts in play beyond the ones drawn.

    The quota decides what a task is *about*; the account and the order decide
    what is *true* of it. A return of cut-to-length goods is refused on the
    category whether or not the ledger happened to draw that policy, and an
    obligation written without it refuses for the wrong reason, so an agent
    that answers correctly is breached against it. Run to a fixpoint: one
    addition can trigger another.
    """
    found = list(chosen)
    for _ in range(len(ledger.rules)):
        added = [
            policy_id for policy_id, rule in ledger.rules.items()
            if policy_id not in found
            and rule.implied_by
            and any(other in rule.implied_by for other in found)
            and matches(subject, rule.implied_when, reference_date)
        ]
        if not added:
            break
        found.extend(added)
    return found[len(chosen):]


def eligible(pool: list[Subject], chosen: list[str], ledger: Ledger,
             reference_date: dt.date) -> list[Subject]:
    """The subjects on which every policy in the combination could bite."""
    found = pool
    for policy_id in chosen:
        selector = ledger.rules[policy_id].subject
        if selector:
            found = [s for s in found if matches(s, selector, reference_date)]
        if not found:
            break
    return found


# --- drawing a combination ----------------------------------------------------


def draw_combination(
    rng: random.Random,
    ledger: Ledger,
    pool: list[Subject],
    open_pool: list[Subject],
    reference_date: dt.date,
    used: Counter[tuple[str, str]],
    owed: dict[str, int],
    tasks_left: int,
    attempts: int = 60,
) -> tuple[list[str], Subject] | None:
    """A policy combination with a subject every one of them could bite on."""
    # A combination that comes up short of the size drawn is retried rather than
    # accepted, until the attempts are nearly gone: taking the short one every
    # time it happens would quietly bend the whole library towards small tasks.
    insist_until = attempts * 9 // 10
    for attempt in range(attempts):
        size = ledger.draw_size(rng)
        solo = ledger.solo_owed()
        needs = ledger.alone_needs()
        slots = ledger.single_slots_left()
        if owed and tasks_left <= _tasks_to_clear(owed, ledger):
            # The batch is running out with policies in it barely drawn. A batch
            # nobody can read a policy's behaviour off is worth less than one
            # whose last few tasks were chosen rather than sampled.
            chosen = [rng.choices(list(owed), weights=list(owed.values()))[0]]
        elif size == 1 and solo:
            # A solo reservation is a debt to a policy that has been masked
            # before, so it is paid before anything else competes for the slot.
            chosen = [rng.choice(solo)]
        elif size == 1 and needs and (not slots or rng.random() < sum(needs.values()) / slots):
            chosen = [rng.choices(list(needs), weights=list(needs.values()))[0]]
        else:
            available = [p for p in ledger.rules if ledger.weight(p)
                         and not (size > 1 and ledger.rules[p].alone)]
            first = _weighted(rng, ledger, available)
            if first is None:
                return None
            chosen = [first]
        if ledger.rules[chosen[0]].alone:
            size = 1
        # Only a precedence-taking policy may draw on the subjects it claims.
        start = pool if ledger.rules[chosen[0]].alone else open_pool
        candidates = eligible(start, chosen, ledger, reference_date)
        if not candidates:
            continue
        while len(chosen) < size:
            joinable = [p for p in ledger.rules if ledger.may_join(chosen, p)]
            # Only keep the ones some subject could still carry alongside.
            narrowed = [
                (p, still) for p in joinable
                if (still := eligible(candidates, [p], ledger, reference_date))
            ]
            if not narrowed:
                break
            policy_id = _weighted(rng, ledger, [p for p, _ in narrowed])
            if policy_id is None:
                break
            chosen.append(policy_id)
            candidates = dict(narrowed)[policy_id]
        if len(chosen) != size and attempt < insist_until:
            continue
        if not ledger.satisfied(chosen) or not candidates:
            continue
        # Some of the desk's post is not about an order: payment terms, a
        # request about another company. Order-bearing subjects outnumber the
        # rest several times over, so left to chance those questions never get
        # asked, and the rule about naming the order number is never seen not to
        # apply. Where the combination can be carried without one, sometimes it is.
        orderless = [s for s in candidates if s.order is None]
        if orderless and rng.random() < ledger.account_level_share:
            candidates = orderless
        # A situation that drags in more than a task can hold is answered by
        # picking a different order or account, never by dropping the policy:
        # the dropped one would still be true, and the obligation would be wrong.
        viable = [
            (s, extra) for s in candidates
            if len(chosen) + len(extra := implied(s, chosen, ledger, reference_date))
            <= MAX_SUBSTANTIVE
        ]
        if not viable:
            continue
        # An implied policy is never dropped, so the only way to keep one from
        # running past its quota is to find a situation that does not summon it.
        # A preference, not a rule: where every remaining subject summons it, the
        # overrun is the right answer and the ledger reports it.
        unspent = [(s, e) for s, e in viable
                   if all(ledger.remaining(policy_id) for policy_id in e)]
        viable = unspent or viable
        # Prefer a subject the batch has leaned on least, so no two tasks share
        # a fact pattern more often than the data forces.
        fewest = min(used[s.key] for s, _ in viable)
        subject, extra = rng.choice([(s, e) for s, e in viable if used[s.key] == fewest])
        return chosen + extra, subject
    return None


# The sweep starts this much earlier than the arithmetic alone would say. A
# swept task pays off one owed exposure reliably and its second only if the data
# allows, so starting exactly on the estimate leaves a few policies short. 1.6
# clears the floor for every policy while still leaving the batch its shape;
# larger values buy nothing and flatten the three-policy tasks away.
SWEEP_MARGIN = 1.6


def _tasks_to_clear(owed: dict[str, int], ledger: Ledger) -> int:
    """How many tasks the outstanding exposures need, at the ledger's own mean.

    Counting one task per exposure would have the sweep running from the first
    draw to the last, and a batch made entirely of swept single-policy tasks is
    not the batch the size quotas describe. A task carries 1.8 substantive
    policies on average, and ordinary draws pay off the debt too.
    """
    total = sum(size * count for size, count in ledger.sizes.items())
    tasks = sum(ledger.sizes.values())
    mean = (total / tasks) if tasks else 1.0
    return math.ceil(sum(owed.values()) / max(mean, 1.0) * SWEEP_MARGIN)


def _weighted(rng: random.Random, ledger: Ledger, available: list[str]) -> str | None:
    """Pick by how much quota is left, so the batch consumes proportionally."""
    weights = [ledger.weight(p) for p in available]
    if not available or not sum(weights):
        return None
    return rng.choices(available, weights=weights)[0]


# --- drafting -----------------------------------------------------------------


# Handed to each call so the batch varies rather than each email being written
# to the model's own default. Weights, not a cycle: a rota would show up as a
# pattern in the finished library.
VARIANT_COUNTS = ((2, 30), (3, 40), (4, 30))
REGISTERS = (
    "neutral and businesslike",
    "chatty, the way somebody writes to a supplier they speak to every week",
    "brisk and to the point, no small talk",
    "mildly fed up, though still perfectly civil",
    "careful and slightly formal",
)
GREETINGS = ((True, 75), (False, 25))
# Emails that all begin the same way read as one hand, however much the middles
# differ, so the opening move is dealt out rather than left to the model's
# habit.
OPENINGS = (
    "Open with the order number and what is wrong with it.",
    "Open with the problem in plain terms and come to the order number after.",
    "Open with a line of context — what the goods are for, what is happening on "
    "site — before the request.",
    "Open with the question itself, and fill in the detail underneath.",
    "Open by referring back to something earlier: a call, a previous order, an "
    "email they think they sent.",
    "Open with an apology for chasing, then the request.",
)

# PPE returnability turns on whether the packaging was opened, which no record
# holds: it is a fact of the return event and only the customer knows it. So a
# return of PPE is written in one of two shapes, and which one is a decision
# taken here rather than left to the model to resolve case by case.
PPE_CATEGORIES = {"ppe", "safety-ppe"}
RETURN_FAMILY = {"returns-window", "restocking-fee", "refund-method",
                 "non-returnable-categories"}
PPE_SHAPES = (
    ("stated-opened", 30, """This customer says plainly that the packaging has been opened. Put
`packaging_condition: opened` in the facts and have every variant say so in its
own words. The obligations refuse the return, naming the PPE category as the
reason, and no goodwill credit is offered as a substitute for it."""),
    ("stated-unopened", 30, """\
This customer says plainly that the packaging is unopened and the goods are as
delivered. Put `packaging_condition: unopened` in the facts and have every
variant say so in its own words. The category therefore does not bar the return,
and the obligations run the ordinary returns and restocking mechanics."""),
    ("silent", 40, """\
This customer says nothing about the state of the packaging, and you must not
make them mention it. Put `packaging_condition: not stated by the customer` in
the facts. Because nothing in the records settles it either, the obligations
require the reply to ask the customer to confirm the packaging condition before
the return is either accepted or refused, and forbid it doing either outright.
Any window or fee mechanics are stated conditionally: "the timing does not bar
the return" and "if the return proceeds", never "the return is accepted"."""),
)


def ppe_shape(rng: random.Random, subject: Subject, chosen: list[str]) -> str:
    """The packaging instruction, where PPE is being sent back."""
    if not subject.order or not (set(chosen) & RETURN_FAMILY):
        return ""
    if not {line.category for line in subject.order.lines} & PPE_CATEGORIES:
        return ""
    shape = rng.choices([s for s, _, _ in PPE_SHAPES],
                        weights=[w for _, w, _ in PPE_SHAPES])[0]
    return "\n\n" + dict((s, t) for s, _, t in PPE_SHAPES)[shape]


def draft_one(
    provider: Any,
    task_id: str,
    subject: Subject,
    chosen: list[str],
    policies: dict[str, Policy],
    ledger: Ledger,
    reference_date: str,
    style: str,
) -> dict[str, Any] | None:
    policy_block = "\n".join(
        f"- {policy_id}: {policies[policy_id].statement}"
        + (f"\n  For the obligation: {ledger.rules[policy_id].obligation_hint}"
           if ledger.rules[policy_id].obligation_hint else "")
        for policy_id in chosen
    )
    order = subject.order
    prompt = (
        f"TODAY'S DATE\n{reference_date}\n\n"
        f"ACCOUNT RECORD\n{json.dumps(subject.account.as_record(), indent=2)}\n\n"
        f"ORDER RECORD\n"
        f"{json.dumps(order.as_record(), indent=2) if order else 'none - write a general enquiry'}"
        f"\n\nPOLICIES IN PLAY (use these ids verbatim in expected_obligations)\n{policy_block}"
        f"\n\nHOW TO WRITE THIS ONE\n{style}"
    )
    completion = provider.complete(SYSTEM_PROMPT, [Message(role="user", text=prompt)], tools=None)
    parsed = _parse(completion.text)
    if parsed is None:
        print(f"rejected {task_id}: model output was not parseable JSON", file=sys.stderr)
        return None

    applicable = list(chosen)
    obligations = list(parsed.get("expected_obligations") or [])
    if order is not None and ledger.style_with_order:
        applicable.append(ledger.style_with_order)
        obligations.append({
            "policy": ledger.style_with_order,
            "obligation": f"{order.id} named in the first sentence.",
        })
    if ledger.style_always:
        applicable.append(ledger.style_always)
        obligations.append({
            "policy": ledger.style_always,
            "obligation": (
                f'ends "Fenmoor Supplies order desk" then {subject.account.id}, nothing after.'
            ),
        })

    entry: dict[str, Any] = {
        "id": task_id,
        "account_id": subject.account.id,
        "applicable_policies": applicable,
        "facts": parsed.get("facts"),
        "variants": parsed.get("variants"),
        "expected_obligations": obligations,
    }
    if order is not None:
        entry["order_id"] = order.id
    return entry


def _parse(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    for candidate in ([text[start : end + 1]] if start != -1 and end > start else []) + [text]:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and "variants" in parsed:
            return parsed
    return None


# --- reporting ----------------------------------------------------------------


def report_ledger(ledger: Ledger, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'policy':<28}{'quota':>6}{'spent':>7}{'left':>6}   solo")
    for policy_id, rule in sorted(ledger.rules.items()):
        solo = ""
        if rule.solo:
            solo = f"{ledger.solo_spent[policy_id]}/{rule.solo}"
        print(f"  {policy_id:<28}{rule.quota:>6}{ledger.spent[policy_id]:>7}"
              f"{ledger.remaining(policy_id):>6}   {solo}")
    total_quota = sum(r.quota for r in ledger.rules.values())
    print(f"  {'total exposures':<28}{total_quota:>6}{sum(ledger.spent.values()):>7}"
          f"{total_quota - sum(ledger.spent.values()):>6}")
    if ledger.sizes:
        shape = ", ".join(
            f"{size}-policy {ledger.sizes_spent[size]}/{total}"
            for size, total in sorted(ledger.sizes.items())
        )
        print(f"  task shapes: {shape}")


# --- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=10, help="how many tasks to draft")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "staging")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--ledger", type=Path, default=None,
        help="exposure ledger: quotas, subject conditions, combination constraints. "
             "scenario/exposure-ledger.yaml is the one that produced the shipped "
             "tasks, and its header documents the format. Omit to draw at random",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=SCENARIO / "data",
        help="where accounts.yaml and orders.yaml live; point at staging to draft "
             "against data not yet committed",
    )
    parser.add_argument(
        "--consumed-from", type=Path, action="append", default=None,
        help="a directory of existing tasks to spend the ledger against; repeatable. "
             "Defaults to scenario/tasks",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="first task number; defaults to one past the highest already in scenario/tasks",
    )
    parser.add_argument(
        "--min-per-policy", type=int, default=1,
        help="draw every policy that still has quota at least this many times in the "
             "batch, if the batch is big enough to. A policy a batch never touches is "
             "a policy the batch says nothing about",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="draw the combinations and report the ledger without calling a model. "
             "Worth doing before a batch: it says whether the data can carry the quotas",
    )
    args = parser.parse_args(argv)
    # Same .env the harness reads, so a role's model and key are configured once.
    load_dotenv()

    policies = load_policies(SCENARIO / "policies.yaml")
    accounts = load_accounts(args.data_dir / "accounts.yaml")
    orders = load_orders(args.data_dir / "orders.yaml", accounts)
    reference_date = yaml.safe_load(
        (args.data_dir / "orders.yaml").read_text(encoding="utf-8")
    )["reference_date"]
    reference_date = str(reference_date)

    ledger = (Ledger.load(args.ledger, policies) if args.ledger
              else Ledger.unconstrained(policies))
    if args.ledger:
        directories = args.consumed_from or [SCENARIO / "tasks"]
        seen = ledger.count_existing([d for d in directories if d.is_dir()])
        print(f"ledger {args.ledger}: spent against {seen} existing task(s) in "
              f"{', '.join(str(d) for d in directories)}")
        report_ledger(ledger, "ledger before this batch")

    provider = None if args.dry_run else build_provider(model_for("agent"))
    rng = random.Random(args.seed)
    pool = subjects(accounts, orders)
    open_pool = unclaimed(pool, ledger, dt.date.fromisoformat(reference_date))
    if len(open_pool) < len(pool):
        print(f"{len(pool) - len(open_pool)} of {len(pool)} subjects are claimed by a "
              "policy that takes precedence; nothing else will be drawn on them")
    used: Counter[tuple[str, str]] = Counter()
    next_id = args.start_id if args.start_id is not None else _next_task_number()
    args.out.mkdir(parents=True, exist_ok=True)

    drafted = 0
    rejected = 0
    offset = 0
    batch: Counter[str] = Counter()
    # A rejected draft is waste, not output, so it does not count against the
    # batch. The attempt cap is only there so a systematically failing prompt
    # stops rather than spending a whole budget on it.
    attempts_allowed = args.count * 3
    while drafted < args.count and drafted + rejected < attempts_allowed:
        # Policies with quota left that this batch has not drawn often enough.
        # The pilot gate reads each policy's behaviour off the batch, so one the
        # batch barely touches is one the batch says little about.
        owed = {p: args.min_per_policy - batch[p] for p in ledger.rules
                if ledger.weight(p) and batch[p] < args.min_per_policy}
        drawn = draw_combination(
            rng, ledger, pool, open_pool, dt.date.fromisoformat(reference_date), used,
            owed, args.count - (drafted + rejected),
        )
        if drawn is None:
            print("no combination left that the data can carry; stopping", file=sys.stderr)
            break
        chosen, subject = drawn
        # No truncation here. Whatever the draw returned includes the policies
        # the situation itself put in play, and cutting one off would leave an
        # obligation that is wrong rather than a task that is smaller.
        task_id = f"task-{next_id + offset:04d}"
        if args.dry_run:
            offset += 1
            ledger.record(chosen)
            used[subject.key] += 1
            batch.update(chosen)
            drafted += 1
            print(f"would draft {task_id}  {subject.account.id}"
                  f"{'/' + subject.order.id if subject.order else ''}  {', '.join(chosen)}")
            continue
        variants = rng.choices([c for c, _ in VARIANT_COUNTS],
                               weights=[w for _, w in VARIANT_COUNTS])[0]
        style = (
            f"Write exactly {variants} variants.\n"
            f"Register: {rng.choice(REGISTERS)}.\n"
            f"{rng.choice(OPENINGS)}\n"
            + ("This customer signs off with a name and opens with some greeting.\n"
               if rng.choices([g for g, _ in GREETINGS], weights=[w for _, w in GREETINGS])[0]
               else "This customer writes with no greeting and no sign-off, straight in.\n")
            + rng.choice([
                "Keep it short: a couple of lines.",
                "A medium-length note, three or four sentences.",
                "A longer one: give some context around the request, a paragraph or two.",
                "A longer one, and let them mention something incidental that no policy "
                "turns on.",
            ])
        )
        style += ppe_shape(rng, subject, chosen)
        entry = draft_one(provider, task_id, subject, chosen, policies, ledger,
                          reference_date, style)
        out_path = args.out / f"{task_id}.yaml"
        if entry is not None:
            # One task per file, named for its id: the same shape scenario/tasks/
            # uses, so a reviewed task moves across with `mv` and nothing else.
            try:
                build_task(out_path, entry, policies, accounts, orders)
            except ScenarioError as error:
                print(f"rejected {task_id}: {error}", file=sys.stderr)
                entry = None
        if entry is None:
            rejected += 1
            continue
        out_path.write_text(
            "# Drafted by tools/generate_tasks.py. Read this before moving it\n"
            "# into scenario/tasks/.\n"
            + yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=88),
            encoding="utf-8",
        )
        ledger.record(chosen)
        used[subject.key] += 1
        batch.update(chosen)
        drafted += 1
        offset += 1  # only a task that was actually written takes an id
        print(f"drafted {task_id}  {subject.account.id}"
              f"{'/' + subject.order.id if subject.order else ''}  {', '.join(chosen)}")

    print(f"\nwrote {drafted} task file(s) to {args.out}; {rejected} rejected")
    if drafted < args.count:
        print(f"stopped {args.count - drafted} short of the batch: too many drafts were "
              f"discarded to keep going", file=sys.stderr)
    if args.ledger:
        report_ledger(ledger, "ledger after this batch")
    return 0


def _next_task_number() -> int:
    highest = 0
    for path in (SCENARIO / "tasks").rglob("*.yaml"):
        for match in re.finditer(r"task-(\d{4})", path.read_text(encoding="utf-8")):
            highest = max(highest, int(match.group(1)))
    return highest + 1


if __name__ == "__main__":
    raise SystemExit(main())
