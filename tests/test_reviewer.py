"""The reviewer decides the metric, so its edges matter more than its middle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memco_harness.reviewer import Reviewer
from memco_harness.scenario import load_scenario

from .mocks import ScriptedProvider

SCENARIO_ROOT = Path(__file__).resolve().parent.parent / "scenario"


@pytest.fixture(scope="module")
def scenario():
    return load_scenario(SCENARIO_ROOT)


@pytest.fixture
def task(scenario):
    return next(task for task in scenario.tasks if len(task.applicable_policies) >= 2)


def review_with(scenario, task, response):
    provider = ScriptedProvider(default=response)
    reviewer = Reviewer(
        provider=provider,
        policies=scenario.policies,
        reference_date=scenario.reference_date,
    )
    review = reviewer.review(
        task,
        task.variants[0],
        "Draft that ignores everything the desk knows.",
        scenario.account_for(task),
        scenario.order_for(task),
    )
    return review, provider


def test_seeded_breaches_are_surfaced(scenario, task):
    seeded = list(task.applicable_policies)
    response = json.dumps(
        {
            "breaches": [{"policy": policy, "note": f"missed {policy}"} for policy in seeded],
            "corrected_draft": "The corrected reply.",
        }
    )
    review, _ = review_with(scenario, task, response)
    assert [breach.policy for breach in review.breaches] == seeded
    assert review.breach_count == len(seeded)
    assert review.corrected_draft == "The corrected reply."
    assert not review.failed


def test_the_id_is_matched_however_the_reviewer_decorates_it(scenario, task):
    """The prompt heads each policy `[its-id] Its title`, so the reviewer copies
    the brackets into its answer. Matching exactly meant the breach was dropped
    as out of scope and the episode recorded as a faultless reply: 41 verdicts
    deleted from one hundred-task run, and eight spotless control episodes that
    this scenario cannot produce. An id is content a model retypes, and no
    amount of decoration on it names a different policy."""
    seeded = list(task.applicable_policies)[:2]
    decorated = [f"[{seeded[0]}]", f'  "{seeded[1].upper()}" ']
    response = json.dumps({
        "breaches": [{"policy": policy, "note": "wrong"} for policy in decorated],
        "corrected_draft": "The corrected reply.",
    })
    review, _ = review_with(scenario, task, response)
    assert [b.policy for b in review.breaches] == seeded, "the canonical ids, not the decoration"
    assert review.dropped == (), "nothing here was out of scope"


def test_a_decorated_duplicate_is_still_one_breach(scenario, task):
    policy = task.applicable_policies[0]
    response = json.dumps({
        "breaches": [{"policy": policy, "note": "a"}, {"policy": f"[{policy}]", "note": "b"}],
        "corrected_draft": "The corrected reply.",
    })
    review, _ = review_with(scenario, task, response)
    assert review.breach_count == 1


def test_out_of_scope_policy_ids_are_dropped(scenario, task):
    response = json.dumps(
        {
            "breaches": [
                {"policy": task.applicable_policies[0], "note": "in scope"},
                {"policy": "sign-off-house-style-but-not-in-scope", "note": "out of scope"},
            ],
            "corrected_draft": "The corrected reply.",
        }
    )
    review, _ = review_with(scenario, task, response)
    assert [breach.policy for breach in review.breaches] == [task.applicable_policies[0]]
    assert review.dropped == ("sign-off-house-style-but-not-in-scope",)


def test_a_duplicated_breach_is_counted_once(scenario, task):
    policy = task.applicable_policies[0]
    response = json.dumps(
        {
            "breaches": [
                {"policy": policy, "note": "first"},
                {"policy": policy, "note": "again"},
            ],
            "corrected_draft": "The corrected reply.",
        }
    )
    review, _ = review_with(scenario, task, response)
    assert review.breach_count == 1


def test_json_inside_a_code_fence_is_accepted(scenario, task):
    payload = json.dumps({"breaches": [], "corrected_draft": "Fine as drafted."})
    review, _ = review_with(scenario, task, f"Here you are:\n```json\n{payload}\n```\n")
    assert review.breach_count == 0
    assert review.corrected_draft == "Fine as drafted."


def test_unparseable_output_is_retried_once_then_recorded_as_failed(scenario, task):
    review, provider = review_with(scenario, task, "I am afraid I cannot do that.")
    assert review.failed
    assert review.breach_count == 0
    assert "not parseable" in review.error
    assert review.raw == "I am afraid I cannot do that.", "keep the raw output to diagnose it"
    assert len(provider.calls) == 2, "the reviewer should retry exactly once"


def test_a_retry_that_parses_is_accepted(scenario, task):
    good = json.dumps({"breaches": [], "corrected_draft": "Second time lucky."})
    answers = iter(["not json at all", good])

    provider = ScriptedProvider(default=lambda system, messages: next(answers))
    reviewer = Reviewer(
        provider=provider,
        policies=scenario.policies,
        reference_date=scenario.reference_date,
    )
    review = reviewer.review(
        task,
        task.variants[0],
        "A draft.",
        scenario.account_for(task),
        scenario.order_for(task),
    )
    assert not review.failed
    assert review.corrected_draft == "Second time lucky."


def test_the_reviewer_sees_the_policies_and_the_agent_never_does(scenario, task):
    response = json.dumps({"breaches": [], "corrected_draft": "Fine."})
    _, provider = review_with(scenario, task, response)
    prompt = provider.calls[0]["messages"][0].text
    for policy_id in task.applicable_policies:
        assert policy_id in prompt
        assert scenario.policies[policy_id].statement in prompt
        assert scenario.policies[policy_id].check_hint in prompt
    for obligation in task.expected_obligations:
        assert obligation.obligation in prompt


def test_the_reviewer_is_told_the_scenario_date_not_the_wall_clock(scenario, task):
    response = json.dumps({"breaches": [], "corrected_draft": "Fine."})
    _, provider = review_with(scenario, task, response)
    assert scenario.reference_date in provider.calls[0]["system"]


def test_only_policies_in_scope_reach_the_prompt(scenario, task):
    """The reviewer's rubric is scoped, or every episode would be judged on all 29."""
    response = json.dumps({"breaches": [], "corrected_draft": "Fine."})
    _, provider = review_with(scenario, task, response)
    prompt = provider.calls[0]["messages"][0].text
    out_of_scope = set(scenario.policies) - set(task.applicable_policies)
    assert not [policy_id for policy_id in out_of_scope if policy_id in prompt]
