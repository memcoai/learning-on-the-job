"""The statistics of a paired run.

These are the numbers the live line prints and the report quotes, so they are
worth checking against arithmetic that can be done by hand rather than only
against the code that produces them.
"""

from __future__ import annotations

from memco_harness.pairing import (
    ADVANTAGE_FLOOR,
    Interval,
    PairStats,
    compliance,
    pooled,
    running_intervals,
    t_critical,
)


def test_the_t_table_covers_the_range_a_run_will_ask_for():
    assert t_critical(1) > 12
    assert t_critical(9) == 2.262
    assert t_critical(30) == 2.042
    # Between tabulated rows it interpolates rather than jumping.
    assert 2.000 < t_critical(50) < 2.021
    # Far out it settles on the normal value, which is what the table converges to.
    assert t_critical(5000) == 1.960


# --- compliance ----------------------------------------------------------------


def test_compliance_is_the_share_of_what_applied_that_was_met():
    assert compliance(breaches=1, applicable=4) == 75.0
    assert compliance(breaches=0, applicable=2) == 100.0
    assert compliance(breaches=3, applicable=3) == 0.0


def test_a_task_with_nothing_applicable_cannot_be_got_wrong():
    """The shipped scenario has no such task; the guard keeps a scenario that
    grows one from dividing by zero halfway through somebody's run."""
    assert compliance(breaches=0, applicable=0) == 100.0


def test_pooling_weighs_a_task_by_how_much_was_at_stake():
    """Six obligations met out of eight is 75%, whichever tasks they came from.
    Averaging the two tasks' own percentages would give 62.5% instead, letting
    the two-policy task count for as much as the six-policy one."""
    assert pooled(breaches=[2, 0], applicable=[2, 6]) == 75.0
    per_task = (compliance(2, 2) + compliance(0, 6)) / 2
    assert per_task == 50.0, "which is the number pooling exists to avoid"


def test_a_pooled_window_moves_in_small_steps_not_in_jumps():
    """The reason for pooling: on ten tasks of four policies the denominator is
    forty, so one breach moves the line 2.5 points rather than 10."""
    breaches, applicable = [0] * 10, [4] * 10
    assert pooled(breaches, applicable) == 100.0
    breaches[3] = 1
    assert pooled(breaches, applicable) == 97.5


# --- the paired difference -----------------------------------------------------


def test_a_difference_is_memory_minus_control_so_positive_means_memory_won():
    stats = PairStats()
    stats.add(memory=1, control=3, applicable=4)  # 75% against 25%
    stats.add(memory=0, control=1, applicable=2)  # 100% against 50%
    assert stats.differences == [50.0, 50.0]
    assert stats.pairs == 2


def test_the_same_count_of_breaches_is_a_different_difference_on_different_tasks():
    """The whole reason the measure changed: one breach out of two is not the
    same failure as one out of six, and a count cannot tell them apart."""
    narrow = PairStats()
    narrow.add(memory=1, control=2, applicable=2)
    wide = PairStats()
    wide.add(memory=1, control=2, applicable=6)
    assert narrow.differences[0] == 50.0
    assert round(wide.differences[0], 4) == round(100 / 6, 4)


def test_the_interval_is_undefined_until_there_are_two_pairs():
    stats = PairStats()
    assert stats.interval() is None
    stats.add(memory=1, control=2, applicable=4)
    assert stats.interval() is None, "one difference has no spread to measure"
    stats.add(memory=1, control=2, applicable=4)
    assert stats.interval() is not None


def test_identical_differences_give_a_point_interval_rather_than_a_crash():
    """No spread is not missing data: it is a genuine, if unlikely, answer."""
    stats = PairStats()
    for _ in range(5):
        stats.add(memory=1, control=3, applicable=4)
    assert stats.interval() == Interval(mean=50.0, low=50.0, high=50.0)


def test_the_interval_matches_the_arithmetic_done_by_hand():
    stats = PairStats()
    # Four policies a task, memory fixed at one breach, so d = 50, 0, 25, 25, 25.
    for control in (3, 1, 2, 2, 2):
        stats.add(memory=1, control=control, applicable=4)
    interval = stats.interval()
    assert interval is not None
    assert round(interval.mean, 4) == 25.0
    # sd = 17.678, se = 7.906, t(4) = 2.776 -> margin 21.947
    assert round(interval.low, 2) == 3.05
    assert round(interval.high, 2) == 46.95


def test_the_advantage_waits_for_enough_pairs_to_be_worth_qualifying():
    """The line and the column both draw on this: a mean from the first pair, an
    interval only once there are ten of them."""
    series = running_intervals([20.0] * ADVANTAGE_FLOOR)
    assert [half for _, half in series[: ADVANTAGE_FLOOR - 1]] == [None] * 9
    assert series[0][0] == 20.0, "the mean is there from the first pair"
    assert series[-1][1] == 0.0, "ten identical pairs: a real interval, of no width"


def test_control_drift_is_reported_only_once_there_is_enough_to_compare():
    stats = PairStats()
    for _ in range(10):
        stats.add(memory=1, control=3, applicable=4)  # control at 25%
    assert stats.control_drift() is None, "not enough pairs to halve"
    for _ in range(10):
        stats.add(memory=1, control=1, applicable=4)  # control at 75%
    assert stats.control_drift() == 50.0, "the sample got easier, in points"


def test_the_trailing_window_follows_the_recent_pairs_not_the_whole_run():
    stats = PairStats()
    for _ in range(10):
        stats.add(memory=4, control=4, applicable=4)
    for _ in range(10):
        stats.add(memory=0, control=3, applicable=4)
    assert stats.trailing(stats.memory_breaches) == 100.0
    assert stats.trailing(stats.control_breaches) == 25.0
    # The whole run still remembers the bad start.
    assert stats.memory_compliance == 50.0


def test_the_record_survives_a_run_that_stopped_after_one_pair():
    """Stopping early is a supported way to use the tool, so every summary has
    to render at any stop point rather than only at the end."""
    stats = PairStats()
    stats.add(memory=1, control=2, applicable=4)
    record = stats.as_record()
    assert record["pairs"] == 1
    assert record["mean_difference"] == 25.0
    assert record["memory_compliance"] == 75.0
    assert record["control_compliance"] == 50.0
    assert "confidence_interval" not in record


def test_the_record_carries_the_raw_counts_under_the_percentages():
    """Leading with a percentage must not lose the counts underneath it."""
    stats = PairStats()
    for _ in range(4):
        stats.add(memory=1, control=3, applicable=4)
    record = stats.as_record()
    assert record["memory_mean"] == 1.0
    assert record["control_mean"] == 3.0
    assert record["applicable_total"] == 16
