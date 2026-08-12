"""Loading and validating the scenario, plus the agent's read-only lookups.

The scenario is data: policies, tasks, and the account and order records behind
the lookup tools. Nothing here knows about models or memory, which is what lets
you swap in your own domain by editing YAML.

The validator follows the committed files rather than a spec of its own. Where
the two ever disagree, the files are the authority and this module is the thing
that changes. It fails fast, naming the file and the field, and it enforces the
invariants the authored data holds to: the VAT identity on every order, line
quantities summing to the goods value, and a credit check recorded on exactly
those orders that need one. Those checks exist for generated content, which is
where an inconsistency would otherwise slip in unnoticed.

`orders.yaml` declares `reference_date`, the scenario's "today". The runner
passes it to the agent and the reviewer, and every window calculation counts
from it. Nothing in an episode reads the wall clock.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Account",
    "GoodwillCredit",
    "Obligation",
    "Order",
    "OrderLine",
    "Payment",
    "Policy",
    "Scenario",
    "ScenarioError",
    "Task",
    "Variant",
    "build_task",
    "load_accounts",
    "load_orders",
    "load_policies",
    "load_scenario",
]

KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TASK_FILENAME = re.compile(r"^task-\d{4}$")
IDENTIFIER = re.compile(r"\b(?:acc|ord)-\d+\b")
# The trailing digit is required: without it the thousands separator swallows a
# comma that belongs to the sentence, and "£6,000, including VAT" reads as a
# different amount from "£6,000 including VAT". Two variants saying the same
# thing with different punctuation are then rejected as disagreeing on the facts.
AMOUNT = re.compile(r"£\s?\d(?:[\d,]*\d)?(?:\.\d+)?")

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "scenario"

# Vocabularies, taken from the headers of the authored data files.
TIERS = {"A", "B", "C"}
CREDIT_STATUSES = {"ok", "on-hold"}
ACCOUNT_STATUSES = {"trading", "closing"}
ORDER_STATUSES = {"before-picking", "picking", "dispatched", "delivered", "awaiting-stock"}
PAYMENT_METHODS = {"card", "invoice"}
CREDIT_CHECKS = {"completed", "pending"}
CATEGORIES = {
    "consumables",
    "fixings",
    "tools",
    "storage",
    "ppe",
    "safety-ppe",
    "calibration",
    "special-order",
    "cut-to-length",
}

# An order above this total carries a credit check; below it, none is recorded.
# The figure is the credit-check policy's threshold, and it lives here only so
# the validator can check that the two files agree with each other.
CREDIT_CHECK_THRESHOLD = 4800.0

VAT_MULTIPLIER = 1.2
PENNY = 0.005

# Policies per task. The authoring convention is one to three substantive
# policies chosen for the task, to which the house-style pair attaches. A fourth
# substantive one is admitted because a situation can put a policy in play that
# nobody chose: a return of goods that are never returnable answers to the
# category as well as to the window, whatever the task was written for. Refusing
# it here would not make the task smaller, it would make its ground truth wrong,
# so the bound is the outer limit that admits and the validator stays ignorant
# of which policies belong to which family.
MAX_POLICIES_PER_TASK = 6


class ScenarioError(ValueError):
    """A scenario file is malformed or a reference does not resolve."""


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    statement: str
    check_hint: str


@dataclass(frozen=True)
class Variant:
    subject: str
    body: str


@dataclass(frozen=True)
class Obligation:
    policy: str
    obligation: str


@dataclass(frozen=True)
class Task:
    id: str
    account_id: str
    applicable_policies: tuple[str, ...]
    facts: dict[str, Any]
    variants: tuple[Variant, ...]
    expected_obligations: tuple[Obligation, ...]
    order_id: str | None = None
    source: str = ""


@dataclass(frozen=True)
class GoodwillCredit:
    """One goodwill credit already granted to an account.

    The goodwill policy counts these over a rolling twelve months, so the desk
    has to be able to see them. Whether a given credit still falls inside the
    window is worked out from the reference date, not recorded here.
    """

    date: str
    amount: float

    def as_record(self) -> dict[str, Any]:
        return {"date": self.date, "amount": self.amount}


@dataclass(frozen=True)
class Account:
    id: str
    name: str
    tier: str  # A | B | C
    credit_status: str  # ok | on-hold
    tenure_years: int
    status: str  # trading | closing
    group: str | None = None
    goodwill_credits: tuple[GoodwillCredit, ...] = ()

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "account_id": self.id,
            "name": self.name,
            "tier": self.tier,
            "credit_status": self.credit_status,
            "tenure_years": self.tenure_years,
            "status": self.status,
        }
        if self.group:
            record["group"] = self.group
        if self.goodwill_credits:
            record["goodwill_credits"] = [credit.as_record() for credit in self.goodwill_credits]
        return record


@dataclass(frozen=True)
class OrderLine:
    description: str
    category: str
    quantity: int
    unit_price_ex_vat: float
    backorder_working_days: int | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "description": self.description,
            "category": self.category,
            "quantity": self.quantity,
            "unit_price_ex_vat": self.unit_price_ex_vat,
        }
        if self.backorder_working_days is not None:
            record["backorder_working_days"] = self.backorder_working_days
        return record


@dataclass(frozen=True)
class Payment:
    amount_inc_vat: float
    settled: bool
    date: str

    def as_record(self) -> dict[str, Any]:
        return {
            "amount_inc_vat": self.amount_inc_vat,
            "settled": self.settled,
            "date": self.date,
        }


@dataclass(frozen=True)
class Order:
    id: str
    account_id: str
    status: str
    payment_method: str
    goods_value_ex_vat: float
    delivery_charge_ex_vat: float
    total_inc_vat: float
    lines: tuple[OrderLine, ...]
    delivered_date: str = ""
    estimated_delivery: str = ""
    carrier: str = ""
    credit_check: str = ""
    payments: tuple[Payment, ...] = ()

    def as_record(self) -> dict[str, Any]:
        """The order as the lookup tool reports it: the authored fields, as authored.

        Nothing is derived here. Whether a delivery is inside a return window,
        or settled payments clear a release threshold, is the agent's to work
        out from these figures and the reference date.
        """
        record: dict[str, Any] = {
            "order_id": self.id,
            "account_id": self.account_id,
            "status": self.status,
            "payment_method": self.payment_method,
            "goods_value_ex_vat": self.goods_value_ex_vat,
            "delivery_charge_ex_vat": self.delivery_charge_ex_vat,
            "total_inc_vat": self.total_inc_vat,
        }
        for key, value in (
            ("delivered_date", self.delivered_date),
            ("estimated_delivery", self.estimated_delivery),
            ("carrier", self.carrier),
            ("credit_check", self.credit_check),
        ):
            if value:
                record[key] = value
        if self.payments:
            record["payments"] = [payment.as_record() for payment in self.payments]
        record["lines"] = [line.as_record() for line in self.lines]
        return record


@dataclass(frozen=True)
class Scenario:
    policies: dict[str, Policy]
    tasks: tuple[Task, ...]
    accounts: dict[str, Account]
    orders: dict[str, Order]
    reference_date: str

    # --- the agent's read-only tools -----------------------------------------

    def lookup_account(self, account_id: str) -> dict[str, Any]:
        account = self.accounts.get(account_id.strip())
        if account is None:
            return {"error": f"no account with id {account_id!r}"}
        return account.as_record()

    def lookup_order(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(order_id.strip())
        if order is None:
            return {"error": f"no order with id {order_id!r}"}
        return order.as_record()

    # --- helpers --------------------------------------------------------------

    def policy(self, policy_id: str) -> Policy:
        return self.policies[policy_id]

    def account_for(self, task: Task) -> Account:
        return self.accounts[task.account_id]

    def order_for(self, task: Task) -> Order | None:
        return self.orders.get(task.order_id) if task.order_id else None


# --- loading ------------------------------------------------------------------


def load_scenario(root: Path | str | None = None) -> Scenario:
    """Load and validate the scenario. Raises `ScenarioError` with file and field."""
    base = Path(root) if root is not None else DEFAULT_ROOT
    policies = _load_policies(base / "policies.yaml")
    accounts = _load_accounts(base / "data" / "accounts.yaml")
    orders, reference_date = _load_orders_file(base / "data" / "orders.yaml", accounts)
    tasks = _load_tasks(base / "tasks", policies, accounts, orders)
    if not tasks:
        raise ScenarioError(f"{base / 'tasks'}: no tasks found")
    return Scenario(
        policies=policies,
        tasks=tasks,
        accounts=accounts,
        orders=orders,
        reference_date=reference_date,
    )


def load_policies(path: Path | str) -> dict[str, Policy]:
    """Load just the policy set. Used by `tools/generate_tasks.py`."""
    return _load_policies(Path(path))


def load_accounts(path: Path | str) -> dict[str, Account]:
    """Load just the accounts. Used by `tools/generate_tasks.py`."""
    return _load_accounts(Path(path))


def load_orders(path: Path | str, accounts: dict[str, Account]) -> dict[str, Order]:
    """Load just the orders. Used by `tools/generate_tasks.py`."""
    return _load_orders_file(Path(path), accounts)[0]


def build_task(
    path: Path,
    entry: Any,
    policies: dict[str, Policy],
    accounts: dict[str, Account],
    orders: dict[str, Order],
) -> Task:
    """Validate one task mapping. Lets the generator check its own output.

    The one-task-per-file and filename rules are not checked here: they are
    properties of a task directory, not of a candidate mapping.
    """
    return _build_task(path, entry, policies, accounts, orders)


# --- primitives ---------------------------------------------------------------


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ScenarioError(f"{path}: file not found")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: invalid YAML: {exc}") from exc


def _require(path: Path, where: str, record: Any, key: str, kind: type = str) -> Any:
    if not isinstance(record, dict):
        raise ScenarioError(f"{path}: {where}: expected a mapping, found {type(record).__name__}")
    if key not in record or record[key] in (None, ""):
        raise ScenarioError(f"{path}: {where}: missing required field {key!r}")
    value = record[key]
    if kind is str and not isinstance(value, str):
        value = _as_text(value)
    elif kind is not str and not isinstance(value, kind):
        raise ScenarioError(f"{path}: {where}: field {key!r} must be {kind.__name__}")
    return value.strip() if isinstance(value, str) else value


def _as_text(value: Any) -> str:
    """YAML turns bare ISO dates into `date` objects; put them back as they were written."""
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _one_of(path: Path, where: str, key: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ScenarioError(
            f"{path}: {where}: {key} must be one of {', '.join(sorted(allowed))}, not {value!r}"
        )
    return value


def _number(path: Path, where: str, entry: dict, key: str) -> float:
    value = entry.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScenarioError(f"{path}: {where}: {key!r} must be a number")
    return float(value)


def _optional_int(path: Path, where: str, entry: dict, key: str) -> int | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScenarioError(f"{path}: {where}: {key!r} must be a whole number")
    return value


def _optional_text(entry: dict, key: str) -> str:
    value = entry.get(key)
    return "" if value is None else _as_text(value).strip()


# --- policies -----------------------------------------------------------------


def _load_policies(path: Path) -> dict[str, Policy]:
    data = _read_yaml(path)
    entries = data.get("policies") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ScenarioError(f"{path}: expected a non-empty 'policies' list")

    policies: dict[str, Policy] = {}
    for index, entry in enumerate(entries):
        where = f"policies[{index}]"
        policy_id = _require(path, where, entry, "id")
        if not KEBAB.match(policy_id):
            raise ScenarioError(f"{path}: {where}: id {policy_id!r} must be kebab-case")
        if policy_id in policies:
            raise ScenarioError(f"{path}: {where}: duplicate policy id {policy_id!r}")
        policies[policy_id] = Policy(
            id=policy_id,
            title=_require(path, where, entry, "title"),
            statement=" ".join(_require(path, where, entry, "statement").split()),
            check_hint=" ".join(_require(path, where, entry, "check_hint").split()),
        )
    return policies


# --- accounts -----------------------------------------------------------------


def _load_accounts(path: Path) -> dict[str, Account]:
    data = _read_yaml(path)
    entries = data.get("accounts") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ScenarioError(f"{path}: expected a non-empty 'accounts' list")

    accounts: dict[str, Account] = {}
    for index, entry in enumerate(entries):
        where = f"accounts[{index}]"
        account_id = _require(path, where, entry, "id")
        if account_id in accounts:
            raise ScenarioError(f"{path}: {where}: duplicate account id {account_id!r}")
        tenure = _require(path, where, entry, "tenure_years", int)
        if tenure < 0:
            raise ScenarioError(f"{path}: {where}: 'tenure_years' must not be negative")
        accounts[account_id] = Account(
            id=account_id,
            name=_require(path, where, entry, "name"),
            tier=_one_of(
                path, where, "tier", _require(path, where, entry, "tier").upper(), TIERS
            ),
            credit_status=_one_of(
                path,
                where,
                "credit_status",
                _require(path, where, entry, "credit_status"),
                CREDIT_STATUSES,
            ),
            tenure_years=tenure,
            status=_one_of(
                path, where, "status", _require(path, where, entry, "status"), ACCOUNT_STATUSES
            ),
            group=_optional_text(entry, "group") or None,
            goodwill_credits=_build_goodwill(path, where, entry.get("goodwill_credits")),
        )
    return accounts


def _build_goodwill(path: Path, where: str, raw: Any) -> tuple[GoodwillCredit, ...]:
    """Goodwill already granted. Absent on most accounts, which have had none."""
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ScenarioError(
            f"{path}: {where}: 'goodwill_credits' must be a non-empty list, or be left out"
        )
    credits: list[GoodwillCredit] = []
    for index, entry in enumerate(raw):
        spot = f"{where}: goodwill_credits[{index}]"
        date = _require(path, spot, entry, "date")
        try:
            dt.date.fromisoformat(date)
        except ValueError as exc:
            raise ScenarioError(
                f"{path}: {spot}: 'date' must be an ISO date, not {date!r}"
            ) from exc
        amount = _number(path, spot, entry, "amount")
        if amount <= 0:
            raise ScenarioError(f"{path}: {spot}: 'amount' must be positive, not {amount}")
        credits.append(GoodwillCredit(date=date, amount=amount))
    return tuple(credits)


# --- orders -------------------------------------------------------------------


def _load_orders_file(
    path: Path, accounts: dict[str, Account]
) -> tuple[dict[str, Order], str]:
    data = _read_yaml(path)
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: expected a mapping with 'reference_date' and 'orders'")

    reference_date = _require(path, "top level", data, "reference_date")
    try:
        dt.date.fromisoformat(reference_date)
    except ValueError as exc:
        raise ScenarioError(
            f"{path}: top level: 'reference_date' must be an ISO date, not {reference_date!r}"
        ) from exc

    entries = data.get("orders")
    if not isinstance(entries, list) or not entries:
        raise ScenarioError(f"{path}: expected a non-empty 'orders' list")

    orders: dict[str, Order] = {}
    for index, entry in enumerate(entries):
        where = f"orders[{index}]"
        order_id = _require(path, where, entry, "id")
        if order_id in orders:
            raise ScenarioError(f"{path}: {where}: duplicate order id {order_id!r}")
        where = f"order {order_id!r}"

        account_id = _require(path, where, entry, "account_id")
        if account_id not in accounts:
            raise ScenarioError(f"{path}: {where}: unknown account_id {account_id!r}")

        goods = _number(path, where, entry, "goods_value_ex_vat")
        delivery = _number(path, where, entry, "delivery_charge_ex_vat")
        total = _number(path, where, entry, "total_inc_vat")
        expected_total = round((goods + delivery) * VAT_MULTIPLIER, 2)
        if abs(expected_total - total) > PENNY:
            raise ScenarioError(
                f"{path}: {where}: 'total_inc_vat' is {total}, but "
                f"({goods} + {delivery}) x {VAT_MULTIPLIER} is {expected_total}"
            )

        lines = _build_lines(path, where, entry.get("lines"))
        line_total = round(sum(line.quantity * line.unit_price_ex_vat for line in lines), 2)
        if abs(line_total - goods) > PENNY:
            raise ScenarioError(
                f"{path}: {where}: lines come to {line_total}, but "
                f"'goods_value_ex_vat' is {goods}"
            )

        credit_check = _optional_text(entry, "credit_check")
        if credit_check:
            _one_of(path, where, "credit_check", credit_check, CREDIT_CHECKS)
        # The two data files have to agree with the credit-check policy about
        # which orders need one, or a task can be authored against a threshold
        # the record does not reflect.
        if (total > CREDIT_CHECK_THRESHOLD) != bool(credit_check):
            expected = "a credit_check" if total > CREDIT_CHECK_THRESHOLD else "no credit_check"
            raise ScenarioError(
                f"{path}: {where}: total_inc_vat {total} requires {expected} "
                f"(the threshold is {CREDIT_CHECK_THRESHOLD})"
            )

        orders[order_id] = Order(
            id=order_id,
            account_id=account_id,
            status=_one_of(
                path, where, "status", _require(path, where, entry, "status"), ORDER_STATUSES
            ),
            payment_method=_one_of(
                path,
                where,
                "payment_method",
                _require(path, where, entry, "payment_method"),
                PAYMENT_METHODS,
            ),
            goods_value_ex_vat=goods,
            delivery_charge_ex_vat=delivery,
            total_inc_vat=total,
            lines=lines,
            delivered_date=_optional_text(entry, "delivered_date"),
            estimated_delivery=_optional_text(entry, "estimated_delivery"),
            carrier=_optional_text(entry, "carrier"),
            credit_check=credit_check,
            payments=_build_payments(path, where, entry.get("payments")),
        )
    return orders, reference_date


def _build_lines(path: Path, where: str, raw: Any) -> tuple[OrderLine, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScenarioError(f"{path}: {where}: 'lines' must be a non-empty list")
    lines: list[OrderLine] = []
    for index, entry in enumerate(raw):
        spot = f"{where}: lines[{index}]"
        lines.append(
            OrderLine(
                description=_require(path, spot, entry, "description"),
                category=_one_of(
                    path, spot, "category", _require(path, spot, entry, "category"), CATEGORIES
                ),
                quantity=_require(path, spot, entry, "quantity", int),
                unit_price_ex_vat=_number(path, spot, entry, "unit_price_ex_vat"),
                backorder_working_days=_optional_int(path, spot, entry, "backorder_working_days"),
            )
        )
    return tuple(lines)


def _build_payments(path: Path, where: str, raw: Any) -> tuple[Payment, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ScenarioError(f"{path}: {where}: 'payments' must be a list")
    payments: list[Payment] = []
    for index, entry in enumerate(raw):
        spot = f"{where}: payments[{index}]"
        settled = entry.get("settled") if isinstance(entry, dict) else None
        if not isinstance(settled, bool):
            raise ScenarioError(f"{path}: {spot}: 'settled' must be true or false")
        payments.append(
            Payment(
                amount_inc_vat=_number(path, spot, entry, "amount_inc_vat"),
                settled=settled,
                date=_require(path, spot, entry, "date"),
            )
        )
    return tuple(payments)


# --- tasks --------------------------------------------------------------------


def _load_tasks(
    directory: Path,
    policies: dict[str, Policy],
    accounts: dict[str, Account],
    orders: dict[str, Order],
) -> tuple[Task, ...]:
    if not directory.is_dir():
        raise ScenarioError(f"{directory}: task directory not found")

    tasks: list[Task] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.rglob("*.yaml")):
        entry = _read_yaml(path)
        if isinstance(entry, list):
            raise ScenarioError(
                f"{path}: holds a list of tasks; one task per file, named for its id"
            )
        task = _build_task(path, entry, policies, accounts, orders)
        if path.stem != task.id:
            raise ScenarioError(f"{path}: holds task {task.id!r}; filename must match the id")
        if not TASK_FILENAME.match(path.stem):
            raise ScenarioError(f"{path}: task files are named task-NNNN.yaml")
        if task.id in seen:
            raise ScenarioError(f"{path}: duplicate task id {task.id!r} (also in {seen[task.id]})")
        seen[task.id] = str(path)
        tasks.append(task)
    return tuple(tasks)


def _build_task(
    path: Path,
    entry: Any,
    policies: dict[str, Policy],
    accounts: dict[str, Account],
    orders: dict[str, Order],
) -> Task:
    task_id = _require(path, "task", entry, "id")
    where = f"task {task_id!r}"

    account_id = _require(path, where, entry, "account_id")
    if account_id not in accounts:
        raise ScenarioError(f"{path}: {where}: unknown account_id {account_id!r}")

    order_id = _optional_text(entry, "order_id") or None
    if order_id is not None:
        if order_id not in orders:
            raise ScenarioError(f"{path}: {where}: unknown order_id {order_id!r}")
        if orders[order_id].account_id != account_id:
            raise ScenarioError(
                f"{path}: {where}: order {order_id!r} belongs to "
                f"{orders[order_id].account_id!r}, not {account_id!r}"
            )

    applicable = entry.get("applicable_policies") or []
    if not isinstance(applicable, list) or not 1 <= len(applicable) <= MAX_POLICIES_PER_TASK:
        raise ScenarioError(
            f"{path}: {where}: 'applicable_policies' must list 1 to "
            f"{MAX_POLICIES_PER_TASK} policy ids"
        )
    for policy_id in applicable:
        if policy_id not in policies:
            raise ScenarioError(f"{path}: {where}: unknown policy {policy_id!r}")
    if len(set(applicable)) != len(applicable):
        raise ScenarioError(f"{path}: {where}: 'applicable_policies' contains a duplicate")

    facts = entry.get("facts") or {}
    if not isinstance(facts, dict) or not facts:
        raise ScenarioError(f"{path}: {where}: 'facts' must be a non-empty mapping")

    variants = _build_variants(path, where, entry.get("variants"))

    obligations = entry.get("expected_obligations") or []
    if not isinstance(obligations, list) or not obligations:
        raise ScenarioError(f"{path}: {where}: 'expected_obligations' must be a non-empty list")
    built: list[Obligation] = []
    for index, obligation in enumerate(obligations):
        spot = f"{where}: expected_obligations[{index}]"
        policy_id = _require(path, spot, obligation, "policy")
        if policy_id not in applicable:
            raise ScenarioError(f"{path}: {spot}: {policy_id!r} is not in applicable_policies")
        built.append(
            Obligation(
                policy=policy_id,
                obligation=" ".join(_require(path, spot, obligation, "obligation").split()),
            )
        )
    covered = {item.policy for item in built}
    missing = [p for p in applicable if p not in covered]
    if missing:
        raise ScenarioError(f"{path}: {where}: no expected obligation for {', '.join(missing)}")

    return Task(
        id=task_id,
        account_id=account_id,
        order_id=order_id,
        applicable_policies=tuple(applicable),
        facts=facts,
        variants=variants,
        expected_obligations=tuple(built),
        source=str(path),
    )


def _build_variants(path: Path, where: str, raw: Any) -> tuple[Variant, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScenarioError(f"{path}: {where}: 'variants' must be a non-empty list")
    variants = tuple(
        Variant(
            subject=_require(path, f"{where}: variants[{index}]", variant, "subject"),
            body=" ".join(_require(path, f"{where}: variants[{index}]", variant, "body").split()),
        )
        for index, variant in enumerate(raw)
    )
    # Variants vary wording only. Identifiers and money amounts are ground
    # truth, so a paraphrase that drops or changes one is a scenario bug.
    for pattern, label in ((IDENTIFIER, "identifier"), (AMOUNT, "amount")):
        sets = [frozenset(pattern.findall(v.body)) for v in variants]
        if len(set(sets)) > 1:
            differing = sorted(set().union(*sets) - set.intersection(*(set(s) for s in sets)))
            raise ScenarioError(
                f"{path}: {where}: variants disagree on {label}(s) "
                f"{', '.join(differing)}; variants may vary wording only"
            )
    return variants
