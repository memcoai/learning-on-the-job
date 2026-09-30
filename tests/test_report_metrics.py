"""Compliance, the paired comparison, and the reconciliation that checks the store.

All pure functions over already-recorded results, so they are tested on
hand-built records rather than by running the loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from memco_harness.memco_client import Insight, Memory, SearchResult
from memco_harness.report import (
    Run,
    deciles,
    pairing_of,
    reconcile,
    render_reconciliation,
    rolling_compliance,
    write_html,
)


def episode(task_id="task-0001", variant=0, breaches=(), told_about=(), lessons=()):
    return {
        "task_id": task_id,
        "variant_index": variant,
        "task_index": 0,
        "review_failed": False,
        "breaches": [{"policy": p, "note": ""} for p in breaches],
        "breach_count": len(breaches),
        "feedback_sent": [
            {"idx": f"memory-{i}-insight-0", "relevant": True, "correct": True,
             "comment": "", "policy": p}
            for i, p in enumerate(told_about)
        ],
        "lessons_retrieved": [{"title": t, "memory_idx": "memory-0", "idx": "memory-0-insight-0"}
                              for t in told_about],
        "lessons_written": list(lessons),
    }


def make_run(records, memory_enabled=True):
    return Run(
        run_id="test-run",
        meta={"memory_enabled": memory_enabled},
        records=tuple(records),
        path=Path("test-run.jsonl"),
    )


# --- reconciliation ------------------------------------------------------------


def stored_result(*titles, times_served=1):
    insights = tuple(
        Insight(idx=f"memory-{i}-insight-0", memory_idx=f"memory-{i}", title=t, content="")
        for i, t in enumerate(titles)
    )
    return SearchResult(
        session_id="session-1",
        memories=tuple(
            Memory(idx=i.memory_idx, insights=(i,), times_served=times_served) for i in insights
        ),
        insights=insights,
    )


def test_reconciliation_reports_what_the_store_kept():
    run = make_run([
        episode(lessons=[
            {"title": "Kept lesson", "query": "q1", "op_id": "create-a-1", "content": ""},
            {"title": "Dropped lesson", "query": "q2", "op_id": "create-a-2", "content": ""},
        ])
    ])
    result = reconcile(
        run, lambda q: stored_result("Kept lesson") if q == "q1" else stored_result()
    )
    assert result.submitted == 2
    assert result.stored == 1
    assert [lesson.title for lesson in result.absent] == ["Dropped lesson"]
    assert "Dropped lesson" in render_reconciliation(result)


def test_consolidation_is_not_inferred_from_the_times_served_count():
    """`times_served` rises on retrieval, not only on a duplicate write.

    Measured on a live server: three consecutive read-only searches for the same
    entry returned 34, 35, 36. Counting merges with it would count our own
    probes, so reconciliation reports presence and leaves merges to the reader.
    """
    run = make_run([
        episode(lessons=[
            {"title": "Seen often", "query": "q", "op_id": "create-a-1", "content": ""}
        ])
    ])
    result = reconcile(run, lambda q: stored_result("Seen often", times_served=36))
    assert result.stored == 1
    assert not hasattr(result, "merged"), "a popularity count cannot count merges"


def test_identical_queries_are_probed_once():
    """Several lessons about one thing share a query; asking twice wastes a call."""
    run = make_run([
        episode(lessons=[
            {"title": "One", "query": "same", "op_id": "a", "content": ""},
            {"title": "Two", "query": "same", "op_id": "b", "content": ""},
        ])
    ])
    asked: list[str] = []

    def search(query):
        asked.append(query)
        return stored_result("One", "Two")

    result = reconcile(run, search)
    assert asked == ["same"]
    assert result.probes == 1
    assert result.stored == 2


def test_reconciliation_paces_between_probes_and_never_writes():
    run = make_run([
        episode(lessons=[
            {"title": "One", "query": "q1", "op_id": "a", "content": ""},
            {"title": "Two", "query": "q2", "op_id": "b", "content": ""},
            {"title": "Three", "query": "q3", "op_id": "c", "content": ""},
        ])
    ])
    slept: list[float] = []
    reconcile(run, lambda q: stored_result(), pace=slept.append, pace_seconds=6)
    assert slept == [6, 6], "between probes only"


def test_an_interrupted_run_is_not_paired_with_a_control_that_finished():
    """The metadata count is the plan, not the outcome.

    A run killed partway through still says it meant to do 16, so pairing on
    that number would overlay ten episodes on sixteen and compare their totals.
    """

    finished = make_run([episode() for _ in range(4)], memory_enabled=False)
    finished.meta.update({"task_count": 4, "seed": None, "all_variants": True,
                          "agent_model": "openai:m"})
    interrupted = make_run([episode() for _ in range(2)])
    interrupted.meta.update({"task_count": 4, "seed": None, "all_variants": True,
                             "agent_model": "openai:m"})
    assert interrupted.task_count == 2, "counts the episodes it produced"
    assert finished.task_count == 4


def test_title_matching_ignores_case_and_spacing():
    run = make_run([
        episode(lessons=[{"title": "  Mixed   Case Title ", "query": "q", "op_id": "a",
                          "content": ""}])
    ])
    result = reconcile(run, lambda q: stored_result("mixed case title"))
    assert result.stored == 1


# --- compliance ----------------------------------------------------------------


def paired(memory_breaches, control_breaches, applicable, meta=None):
    """A paired run as it sits on disk: both arms interleaved in one file."""
    records = []
    for index, (mem, ctl) in enumerate(zip(memory_breaches, control_breaches, strict=True)):
        for arm, count in (("memory", mem), ("control", ctl)):
            record = episode(task_id=f"task-{index:04d}", breaches=("p",) * count)
            record.update({
                "arm": arm,
                "pair_index": index,
                "task_index": index,
                "applicable_policies": [f"p{n}" for n in range(applicable[index])],
            })
            records.append(record)
    return Run(
        run_id="paired-run",
        meta={"memory_enabled": True, "paired": True, **(meta or {})},
        records=tuple(records),
        path=Path("paired-run.jsonl"),
    )


def test_an_arm_is_scored_on_what_applied_not_on_what_went_wrong():
    run = paired([1, 0], [3, 2], [4, 4])
    memory, control = run.as_arm("memory"), run.as_arm("control")
    assert memory.compliance == 87.5, "7 of 8 obligations met"
    assert control.compliance == 37.5, "3 of 8"
    assert memory.mean == 0.5, "the breach counts are still there underneath"


def test_an_episode_that_never_scored_is_left_out_rather_than_counted_as_clean():
    """A failed episode was recorded with no breaches because there was nothing
    else to record. Counting it as a clean reply would flatter its arm."""
    run = paired([2, 0], [2, 0], [4, 4])
    broken = [dict(r) for r in run.records]
    for record in broken:
        if record["arm"] == "memory" and record["pair_index"] == 1:
            record["episode_failed"] = True
    partial = Run(run_id=run.run_id, meta=run.meta, records=tuple(broken), path=run.path)
    memory = partial.as_arm("memory")
    assert memory.compliance == 50.0, "the one scored episode, not a free 100%"
    assert len(memory.scored) == 1


def test_metadata_written_before_the_measure_changed_is_recomputed():
    """Older runs recorded a mean difference in breaches per task, and this
    build reads that field as percentage points. Trusting it would print a
    number that is wrong by whatever the two scales differ by, and the episodes
    are still on disk to be asked again."""
    stale = {"pairs": 2, "mean_difference": 1.5, "memory_mean": 0.5, "control_mean": 2.0}
    run = paired([1, 0], [3, 2], [4, 4], meta={"pairing": stale})
    fresh = pairing_of(run)
    assert fresh["mean_difference"] == 50.0, "percentage points, not breaches"
    assert fresh["memory_compliance"] == 87.5
    # And the split arms carry the recomputed figures, because after the split
    # there is no control arm left to recompute them from.
    assert run.as_arm("memory").meta["pairing"]["mean_difference"] == 50.0


def test_a_current_pairing_record_is_taken_at_its_word():
    current = {"pairs": 2, "mean_difference": 12.0, "memory_compliance": 90.0,
               "control_compliance": 78.0}
    run = paired([1, 0], [3, 2], [4, 4], meta={"pairing": current})
    assert pairing_of(run)["mean_difference"] == 12.0


def test_a_decile_pools_its_bucket_rather_than_averaging_the_tasks():
    """Six obligations met out of eight is 75%, whichever tasks they came from."""
    buckets = deciles([2, 0], [2, 6], buckets=1)
    assert buckets == [(1, 2, 75.0)]


def test_the_trend_line_pools_the_window_it_covers():
    line = rolling_compliance([0, 0, 4], [4, 4, 4], window=2)
    assert line[0] == 100.0, "the first point pools the one task that exists"
    assert line[1] == 100.0
    assert line[2] == 50.0, "four met of eight across the last two tasks"


# --- breaches only the memory arm made -----------------------------------------
#
# A lesson applied outside the conditions it states makes a reply worse than one
# drafted with nothing to go on. Netted into the paired difference it disappears,
# so it is counted on its own.


def two_arms(pairs, told_about=None):
    """A paired run where each arm's breaches are named policy by policy.

    `pairs` is a list of (policies the memory arm breached, policies the blind
    arm breached); `told_about` maps a pair index to the policies a lesson
    retrieved by the memory arm was judged to bear on.
    """
    told_about = told_about or {}
    records = []
    for index, (mem, ctl) in enumerate(pairs):
        for arm, breaches in (("memory", mem), ("control", ctl)):
            record = episode(
                task_id=f"task-{index:04d}",
                breaches=breaches,
                told_about=told_about.get(index, ()) if arm == "memory" else (),
            )
            record.update({
                "arm": arm,
                "pair_index": index,
                "task_index": index,
                "applicable_policies": ["returns-window", "refund-method", "volume-breaks"],
            })
            records.append(record)
    return Run(
        run_id="paired-run",
        meta={"memory_enabled": True, "paired": True},
        records=tuple(records),
        path=Path("paired-run.jsonl"),
    )


def arms(run):
    return run.as_arm("memory"), run.as_arm("control")


def test_the_page_carries_the_result_and_points_at_the_records_for_the_rest(tmp_path):
    """The page answers one question — did memory help. The per-episode analysis
    behind it stays recorded, and the page says where, rather than rendering it:
    a first-time viewer could not tell what the page was trying to say."""
    run = two_arms([(("returns-window",), ())], told_about={0: ("returns-window",)})
    memory, control = arms(run)
    page = write_html(memory, control, tmp_path / "report.html").read_text(encoding="utf-8")
    for gone in ("Breaches the agent had already been told about",
                 "Breaches only the memory arm made",
                 "Per-policy coverage",
                 "materialised policy book"):
        assert gone not in page, f"{gone!r} is no longer part of the report"
    assert ".jsonl" in page and ".pairs.jsonl" in page
    # And the page still ends on the arm table plus that pointer.
    assert page.count("<h2>") == 1, "the learning curve is the only section"


def test_the_pairs_sidecar_is_not_mistaken_for_a_run(tmp_path):
    """A paired run leaves a `<run>.pairs.jsonl` of one line per pair beside its
    episodes. It matched the glob and parsed, held no episodes, and so reported
    a hundred per cent compliance out of nothing — and its name sorts after the
    run's own, so it was what `report` with no argument chose."""
    from memco_harness.report import all_runs, latest_run

    (tmp_path / "run-1.jsonl").write_text(
        json.dumps({"task_id": "task-0001", "arm": "memory", "pair_index": 0,
                    "breach_count": 1, "applicable_policies": ["p1", "p2"]}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run-1.pairs.jsonl").write_text(
        json.dumps({"pair_index": 0, "task_id": "task-0001", "memory_breaches": 1,
                    "control_breaches": 2, "applicable": 2}) + "\n",
        encoding="utf-8",
    )
    assert [run.run_id for run in all_runs(tmp_path)] == ["run-1"]
    assert latest_run(tmp_path).run_id == "run-1"
    assert latest_run(tmp_path).compliance == 50.0, "the episodes, not the sidecar"
