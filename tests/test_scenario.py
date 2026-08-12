"""The scenario is data, so these tests are about the data being sound.

They run against the scenario this repository ships, and against small broken
scenarios built in a temporary directory to check that the validator names the
file and the field it objected to. The fixtures below mirror the authored files'
schema; where the two ever diverge, the authored files are right.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from memco_harness.scenario import MAX_POLICIES_PER_TASK, ScenarioError, load_scenario

SCENARIO_ROOT = Path(__file__).resolve().parent.parent / "scenario"

# The library is grown in stages: seed tasks first, ~60 for a runnable scenario,
# 200-300 for the full convergence run. These bounds tracked the current stage
# and tightened as the library grew; the library is now complete, so they are
# the bounds PLAN set out for it. Five per policy is that floor, and the
# thinnest policy currently sits at eight, so the slack is deliberate: a policy
# dropping to four is a scenario that has lost coverage, not a test to relax.
MIN_TASKS = 200
MIN_TASKS_PER_POLICY = 5


@pytest.fixture(scope="module")
def scenario():
    return load_scenario(SCENARIO_ROOT)


def test_policy_set_is_the_documented_size(scenario):
    assert 25 <= len(scenario.policies) <= 30


def test_task_library_is_at_least_the_seed_set(scenario):
    assert len(scenario.tasks) >= MIN_TASKS


def test_every_policy_touched_by_a_task_has_an_obligation_for_it(scenario):
    counts = Counter(
        policy_id for task in scenario.tasks for policy_id in task.applicable_policies
    )
    thin = {
        policy_id: count for policy_id, count in counts.items() if count < MIN_TASKS_PER_POLICY
    }
    assert not thin, f"policies with too few tasks: {thin}"


def test_each_task_stays_within_the_policies_per_task_bound(scenario):
    for task in scenario.tasks:
        assert 1 <= len(task.applicable_policies) <= MAX_POLICIES_PER_TASK, task.id


def test_every_task_reference_resolves(scenario):
    for task in scenario.tasks:
        assert task.account_id in scenario.accounts, task.id
        if task.order_id:
            assert task.order_id in scenario.orders, task.id
            assert scenario.orders[task.order_id].account_id == task.account_id, task.id


def test_every_applicable_policy_has_an_expected_obligation(scenario):
    for task in scenario.tasks:
        covered = {obligation.policy for obligation in task.expected_obligations}
        assert covered == set(task.applicable_policies), task.id


def test_every_task_has_at_least_two_variants(scenario):
    for task in scenario.tasks:
        assert len(task.variants) >= 2, task.id


def test_the_scenario_has_a_clock(scenario):
    assert scenario.reference_date == "2026-09-14"


def test_vat_and_line_arithmetic_holds_on_every_order(scenario):
    for order in scenario.orders.values():
        expected = round((order.goods_value_ex_vat + order.delivery_charge_ex_vat) * 1.2, 2)
        assert abs(expected - order.total_inc_vat) < 0.005, order.id
        lines = round(sum(line.quantity * line.unit_price_ex_vat for line in order.lines), 2)
        assert abs(lines - order.goods_value_ex_vat) < 0.005, order.id


def test_lookup_tools_answer_and_report_unknown_ids(scenario):
    task = scenario.tasks[0]
    account = scenario.lookup_account(task.account_id)
    assert account["tier"] in {"A", "B", "C"}
    assert isinstance(account["tenure_years"], int)
    assert "error" in scenario.lookup_account("acc-does-not-exist")
    assert "error" in scenario.lookup_order("ord-does-not-exist")


def test_goodwill_history_reaches_the_agent_through_the_lookup(scenario):
    """The goodwill cap is counted over a rolling twelve months, so an account's
    recent credits have to be visible to the desk. The count is the agent's to
    do against the reference date; the record only lists what was granted."""
    with_history = [
        account for account in scenario.accounts.values() if account.goodwill_credits
    ]
    for account in with_history:
        record = scenario.lookup_account(account.id)
        assert record["goodwill_credits"] == [
            {"date": credit.date, "amount": credit.amount} for credit in account.goodwill_credits
        ]
        assert all(credit.amount > 0 for credit in account.goodwill_credits)
    # An account that has had none says nothing rather than saying zero.
    without = next(a for a in scenario.accounts.values() if not a.goodwill_credits)
    assert "goodwill_credits" not in scenario.lookup_account(without.id)


def test_the_order_record_reports_the_authored_fields_and_derives_nothing(scenario):
    record = scenario.lookup_order("ord-3316")
    assert record["total_inc_vat"] == 6000.0
    assert record["credit_check"] == "completed"
    assert [p["settled"] for p in record["payments"]] == [True, False]
    assert record["lines"][0]["category"] == "consumables"
    # Whether the settled payments clear the release threshold is the agent's
    # sum to do, not a field it can read off.
    assert not any("threshold" in key or "release" in key for key in record)


# --- the validator ------------------------------------------------------------

POLICIES = """\
policies:
  - id: only-policy
    title: The only policy
    statement: Always say the thing.
    check_hint: Breach if the draft does not say the thing.
"""

ACCOUNTS = """\
accounts:
  - id: acc-001
    name: Testing Ltd
    tier: B
    credit_status: ok
    tenure_years: 4
    status: trading
"""

ORDERS = """\
reference_date: 2026-09-14

orders:
  - id: ord-001
    account_id: acc-001
    status: before-picking
    payment_method: invoice
    goods_value_ex_vat: 400
    delivery_charge_ex_vat: 20
    total_inc_vat: 504
    lines:
      - description: A box of things
        category: consumables
        quantity: 10
        unit_price_ex_vat: 40
"""

TASK = """\
id: task-0001
account_id: acc-001
order_id: ord-001
applicable_policies: [only-policy]
facts:
  requested_action: ask about order ord-001
variants:
  - subject: A question
    body: Where has order ord-001 got to?
  - subject: Chasing
    body: Any news on order ord-001, please?
expected_obligations:
  - policy: only-policy
    obligation: say the thing
"""


def build_scenario(
    root: Path,
    task: str = TASK,
    policies: str = POLICIES,
    accounts: str = ACCOUNTS,
    orders: str = ORDERS,
    task_name: str = "task-0001.yaml",
) -> Path:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "policies.yaml").write_text(policies, encoding="utf-8")
    (root / "data" / "accounts.yaml").write_text(accounts, encoding="utf-8")
    (root / "data" / "orders.yaml").write_text(orders, encoding="utf-8")
    (root / "tasks" / task_name).write_text(task, encoding="utf-8")
    return root


def test_a_minimal_scenario_loads(tmp_path):
    scenario = load_scenario(build_scenario(tmp_path / "s"))
    assert len(scenario.tasks) == 1
    assert scenario.tasks[0].order_id == "ord-001"
    assert scenario.reference_date == "2026-09-14"


def test_unknown_account_names_the_task(tmp_path):
    broken = TASK.replace("account_id: acc-001", "account_id: acc-999")
    with pytest.raises(ScenarioError) as caught:
        load_scenario(build_scenario(tmp_path / "s", task=broken))
    assert "task 'task-0001'" in str(caught.value)
    assert "acc-999" in str(caught.value)


def test_unknown_policy_is_rejected(tmp_path):
    broken = TASK.replace("[only-policy]", "[no-such-policy]")
    with pytest.raises(ScenarioError, match="no-such-policy"):
        load_scenario(build_scenario(tmp_path / "s", task=broken))


def test_missing_field_names_the_field(tmp_path):
    broken = POLICIES.replace("    check_hint: Breach if the draft does not say the thing.\n", "")
    with pytest.raises(ScenarioError, match="check_hint"):
        load_scenario(build_scenario(tmp_path / "s", policies=broken))


def test_variants_may_not_disagree_on_an_identifier(tmp_path):
    broken = TASK.replace("Any news on order ord-001, please?", "Any news on order ord-002?")
    with pytest.raises(ScenarioError, match="variants may vary wording only"):
        load_scenario(build_scenario(tmp_path / "s", task=broken))


def test_variants_may_not_disagree_on_an_amount(tmp_path):
    broken = TASK.replace(
        "Where has order ord-001 got to?", "Where has order ord-001 got to? We paid £400."
    )
    with pytest.raises(ScenarioError, match="variants may vary wording only"):
        load_scenario(build_scenario(tmp_path / "s", task=broken))


def test_the_same_amount_punctuated_differently_is_the_same_amount(tmp_path):
    """Variants vary wording, and wording includes where the commas fall. An
    amount followed by a comma must not read as a different amount."""
    fine = TASK.replace(
        "Where has order ord-001 got to?", "Where has order ord-001, at £6,000, got to?"
    ).replace("Any news on order ord-001, please?", "Any news on order ord-001 at £6,000?")
    scenario = load_scenario(build_scenario(tmp_path / "s", task=fine))
    assert len(scenario.tasks[0].variants) == 2


def test_an_obligation_outside_applicable_policies_is_rejected(tmp_path):
    broken = TASK + "  - policy: some-other-policy\n    obligation: not in scope\n"
    with pytest.raises(ScenarioError, match="not in applicable_policies"):
        load_scenario(build_scenario(tmp_path / "s", task=broken))


def test_a_task_file_must_be_named_for_its_id(tmp_path):
    with pytest.raises(ScenarioError, match="filename must match the id"):
        load_scenario(build_scenario(tmp_path / "s", task_name="task-0099.yaml"))


def test_a_file_holding_several_tasks_is_rejected(tmp_path):
    grouped = f"- {TASK.replace(chr(10), chr(10) + '  ')}"
    with pytest.raises(ScenarioError, match="one task per file"):
        load_scenario(build_scenario(tmp_path / "s", task=grouped))


def test_broken_vat_arithmetic_is_rejected(tmp_path):
    broken = ORDERS.replace("total_inc_vat: 504", "total_inc_vat: 500")
    with pytest.raises(ScenarioError, match="total_inc_vat"):
        load_scenario(build_scenario(tmp_path / "s", orders=broken))


def test_lines_that_do_not_sum_to_the_goods_value_are_rejected(tmp_path):
    broken = ORDERS.replace("quantity: 10", "quantity: 9")
    with pytest.raises(ScenarioError, match="goods_value_ex_vat"):
        load_scenario(build_scenario(tmp_path / "s", orders=broken))


def test_an_order_over_the_threshold_must_record_a_credit_check(tmp_path):
    broken = (
        ORDERS.replace("goods_value_ex_vat: 400", "goods_value_ex_vat: 5000")
        .replace("total_inc_vat: 504", "total_inc_vat: 6024")
        .replace("unit_price_ex_vat: 40", "unit_price_ex_vat: 500")
    )
    with pytest.raises(ScenarioError, match="requires a credit_check"):
        load_scenario(build_scenario(tmp_path / "s", orders=broken))


def test_an_unknown_line_category_is_rejected(tmp_path):
    broken = ORDERS.replace("category: consumables", "category: gadgets")
    with pytest.raises(ScenarioError, match="category must be one of"):
        load_scenario(build_scenario(tmp_path / "s", orders=broken))


GOODWILL = """\
    goodwill_credits:
      - date: 2026-04-22
        amount: 145
"""


def test_goodwill_credits_load_when_present(tmp_path):
    accounts = ACCOUNTS + GOODWILL
    scenario = load_scenario(build_scenario(tmp_path / "s", accounts=accounts))
    granted = scenario.accounts["acc-001"].goodwill_credits
    assert [(c.date, c.amount) for c in granted] == [("2026-04-22", 145.0)]


def test_a_goodwill_date_that_is_not_a_date_is_rejected(tmp_path):
    accounts = ACCOUNTS + GOODWILL.replace("2026-04-22", "last April")
    with pytest.raises(ScenarioError, match="must be an ISO date"):
        load_scenario(build_scenario(tmp_path / "s", accounts=accounts))


def test_a_goodwill_amount_that_is_not_positive_is_rejected(tmp_path):
    accounts = ACCOUNTS + GOODWILL.replace("amount: 145", "amount: 0")
    with pytest.raises(ScenarioError, match="'amount' must be positive"):
        load_scenario(build_scenario(tmp_path / "s", accounts=accounts))


def test_a_missing_reference_date_is_rejected(tmp_path):
    broken = ORDERS.replace("reference_date: 2026-09-14\n", "")
    with pytest.raises(ScenarioError, match="reference_date"):
        load_scenario(build_scenario(tmp_path / "s", orders=broken))
