"""Reading results back: terminal summary and a self-contained HTML report.

The two surfaces answer different questions, and the split is deliberate.

The HTML report answers one: did memory help. It is the learning curve, which
carries both arms and the gap between them on one axis, and the figures that
read it. Nothing else. It grew a per-episode analysis section for every
question we asked of our own runs, and a first-time viewer could no longer tell
what the page was trying to say. Everything those sections computed is still
computed and still recorded per task in the JSONL, which is where somebody
doing their own analysis, on their own scenario, would start anyway. The page
says so at the end and stops.

The terminal summary is that analysis: deciles, per-policy coverage, breach
notes, the known-versus-novel split, what memory cost. It is read once, by
whoever ran the run, next to the run itself — so it can be as long as the
questions warrant.

One chart, not two. The comparison used to have a second graph of its own, and
a reader meeting the page for the first time had to work out how the two related
before either meant anything. It is the third line on the curve now, in the same
units as the other two, named the same thing the terminal names it.

The framing is deliberately plain. Compliance climbs towards, not to, a hundred;
a small residual is expected, and is what production looks like.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .display import drift_note, preamble
from .display import drift_note as _drift
from .grader_version import GRADER_VERSION
from .pairing import (
    ADVANTAGE_FLOOR,
    TRAILING,
    PairStats,
    compliance,
    pooled,
    running_intervals,
)

__all__ = [
    "Reconciliation",
    "Run",
    "find_control",
    "latest_run",
    "load_run",
    "reconcile",
    "render_reconciliation",
    "render_report",
    "write_html",
]

# The same window the live table pools over, so the line a reader watches during
# the run and the chart they read afterwards are the same measurement.
WINDOW = TRAILING


@dataclass(frozen=True)
class Run:
    run_id: str
    meta: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    path: Path

    @property
    def memory_enabled(self) -> bool:
        return bool(self.meta.get("memory_enabled", True))

    @property
    def arm(self) -> str:
        """What to call this run's own curve. A run reported on its own is often
        the control arm, and labelling it "memory" would misdescribe it."""
        return "memory" if self.memory_enabled else "no memory"

    @property
    def paired(self) -> bool:
        """Both arms in one file, one task at a time. See runner._run_paired."""
        return bool(self.meta.get("paired"))

    def as_arm(self, arm: str) -> Run:
        """This run seen as one of its two arms.

        A paired run holds both arms interleaved in a single file. Everything
        downstream — the curve, the deciles, the per-policy table, the lesson
        book — was written for a run that is one arm, and is still right about
        one arm of a paired run. So the split happens here, once, rather than
        every reader learning about pairing.
        """
        records = tuple(r for r in self.records if r.get("arm", "memory") == arm)
        meta = {**self.meta, "memory_enabled": arm == "memory"}
        # Settled here, because one arm alone cannot work it out: the comparison
        # is a property of the pair, not of either half. Assigned rather than
        # defaulted, so metadata written before the measure changed is replaced
        # with the recomputed figures while both arms are still in hand — after
        # the split there is nothing left to recompute it from.
        meta["pairing"] = pairing_of(self)
        if not meta["pairing"]:
            meta.pop("pairing")
        return Run(run_id=f"{self.run_id}:{arm}", meta=meta, records=records, path=self.path)

    @property
    def seed(self) -> Any:
        return self.meta.get("seed")

    @property
    def task_count(self) -> int:
        """Episodes this run actually produced.

        The metadata's count is what the run set out to do, and an interrupted
        run never gets to correct it. Pairing arms on the planned number would
        overlay a run that stopped early on a control that did not.
        """
        return len(self.records)

    @property
    def scored(self) -> list[dict[str, Any]]:
        """The episodes that actually produced a score.

        An episode that fell over, or whose review could not be read, was
        recorded with a breach count of zero because there was nothing else to
        record. Counting that as a clean reply would flatter whichever arm it
        landed on, so it is left out of every figure rather than quietly
        counted as a pass.
        """
        return [
            record
            for record in self.records
            if not record.get("review_failed") and not record.get("episode_failed")
        ]

    @property
    def breaches(self) -> list[int]:
        return [int(record.get("breach_count", 0)) for record in self.scored]

    @property
    def applicable(self) -> list[int]:
        """How many policies were in play per episode: the denominator."""
        return [len(record.get("applicable_policies", [])) for record in self.scored]

    @property
    def compliance(self) -> float:
        """The share of applicable policies this arm met, over the whole run."""
        return pooled(self.breaches, self.applicable)

    @property
    def mean(self) -> float:
        """Breaches per episode. Still recorded, no longer the headline."""
        counts = self.breaches
        return sum(counts) / len(counts) if counts else 0.0

    @property
    def grader_versions(self) -> set[str]:
        return {str(record.get("grader_version", "?")) for record in self.records}



# --- loading ------------------------------------------------------------------



def pairing_of(run: Run) -> dict[str, Any]:
    """The paired comparison, from the metadata if it is there and from the
    episodes if it is not.

    A run that ends cleanly writes the summary into its metadata. A run that is
    killed outright never gets to, and the numbers would then be missing from
    the report even though every episode that produced them is sitting in the
    JSONL. Recomputing costs nothing and means no way of stopping a run can take
    the headline away from it.

    Metadata written before the measure changed is also recomputed. Those runs
    recorded a mean difference in breaches per task, and this build reads that
    field as percentage points, so trusting it would print a number that is
    wrong by whatever the two scales happen to differ by. The episodes are still
    on disk and still say what happened, so they are asked again.
    """
    memory = [r for r in run.records if r.get("arm") == "memory"]
    control = [r for r in run.records if r.get("arm") == "control"]
    recorded = run.meta.get("pairing")
    if recorded and "memory_compliance" in recorded:
        settled = dict(recorded)
        # Counted from the episodes whenever both arms are in hand, so a run
        # whose metadata predates the count still reports it, and a metadata
        # figure can never disagree with the file it describes.
        if memory and control:
            settled["excluded_pairs"] = _excluded_pairs(memory, control)
        return settled
    if not memory or not control:
        return {}
    stats = PairStats()
    # Only pairs that got both halves in, and only halves that were scored: a
    # half-finished pair is not a measurement of anything.
    by_index = {r["pair_index"]: r for r in control}
    for episode in memory:
        other = by_index.get(episode.get("pair_index"))
        if other is None or _unscored(episode) or _unscored(other):
            continue
        stats.add(
            episode["breach_count"],
            other["breach_count"],
            len(episode.get("applicable_policies", [])),
        )
    if not stats.pairs:
        return {}
    return {**stats.as_record(), "excluded_pairs": _excluded_pairs(memory, control)}


def _excluded_pairs(
    memory: list[dict[str, Any]], control: list[dict[str, Any]]
) -> int:
    """Pairs that answered a task but never scored it.

    Both halves are answered before either is reviewed, so a pair can be
    complete on disk and still be no measurement: one unreadable review leaves
    that arm with a breach count of zero standing in for a verdict nobody
    reached. Reported rather than absorbed, because the tasks were paid for and
    a reader comparing two runs should see that one of them scored fewer of
    them.
    """
    by_index = {r.get("pair_index"): r for r in control}
    return sum(
        1
        for episode in memory
        if (other := by_index.get(episode.get("pair_index"))) is not None
        and (_unscored(episode) or _unscored(other))
    )


def _unscored(record: dict[str, Any]) -> bool:
    return bool(record.get("review_failed") or record.get("episode_failed"))


def load_run(path: Path) -> Run:
    """Load one run from its `.jsonl` path (its `.meta.json` sits beside it)."""
    records = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    meta_path = path.parent / f"{path.stem}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return Run(run_id=meta.get("run_id", path.stem), meta=meta, records=records, path=path)


def all_runs(results_dir: Path) -> list[Run]:
    """Every run in the directory, and nothing that merely looks like one.

    A paired run leaves a `<run>.pairs.jsonl` beside its episodes, one line per
    pair rather than per episode. It matched the glob, it parsed, and it has no
    episodes in it, so it reported a hundred per cent compliance out of nothing
    at all — and because its name sorts after the run's own, it was what `report`
    with no argument picked. A summary that cannot be wrong is not a reassuring
    one.
    """
    return [
        load_run(path)
        for path in sorted(results_dir.glob("*.jsonl"))
        if not path.name.endswith(".pairs.jsonl")
    ]


def latest_run(results_dir: Path) -> Run | None:
    """The most recent run, preferring an arm that had memory on."""
    runs = all_runs(results_dir)
    if not runs:
        return None
    with_memory = [run for run in runs if run.memory_enabled]
    return (with_memory or runs)[-1]


def find_control(results_dir: Path, run: Run) -> Run | None:
    """The matching no-memory arm: same episodes, same models, memory off.

    Same seed and count means the same tasks in the same order with the same
    phrasing, which is what makes the two curves comparable. A memory-off run
    has no control arm of its own: pairing two of them would put an unrelated
    run in the control column, which under `--all-variants` (where no seed
    distinguishes them) would otherwise happen to any two runs of equal length.
    """
    if run.paired:
        # Its control arm is its own: same task, same email, same run. Going
        # looking for a separate run would find a worse comparison than the one
        # already in hand.
        return run.as_arm("control")
    if not run.memory_enabled:
        return None
    candidates = [
        other
        for other in all_runs(results_dir)
        if not other.memory_enabled
        and other.seed == run.seed
        and other.task_count == run.task_count
        and bool(other.meta.get("all_variants")) == bool(run.meta.get("all_variants"))
        and other.meta.get("agent_model") == run.meta.get("agent_model")
        and other.run_id != run.run_id
    ]
    return candidates[-1] if candidates else None


def find_run(results_dir: Path, run_id: str) -> Run | None:
    path = results_dir / f"{run_id}.jsonl"
    return load_run(path) if path.exists() else None


# --- analysis -----------------------------------------------------------------


def rolling_compliance(
    breaches: list[int], applicable: list[int], window: int = WINDOW
) -> list[float]:
    """The trend line: compliance pooled over the trailing window, per task.

    Pooled, not averaged. Over ten tasks the denominator is every obligation
    that applied across them, so a six-policy task weighs more than a two-policy
    one — which is right, because it had more to get wrong. Averaging per-task
    percentages instead would let a two-policy task, whose only possible scores
    are 0, 50 and 100, jerk the line about on its own.

    The first few points are pooled over the tasks that exist. A share of
    obligations met reads correctly from the first task, however few are behind
    it, so unlike an interval it has nothing to wait for.
    """
    out: list[float] = []
    for index in range(len(breaches)):
        start = max(0, index - window + 1)
        out.append(pooled(breaches[start : index + 1], applicable[start : index + 1]))
    return out


def deciles(
    breaches: list[int], applicable: list[int], buckets: int = 10
) -> list[tuple[int, int, float]]:
    """Pooled compliance per bucket, as (first task number, last task number, %)."""
    if not breaches:
        return []
    buckets = min(buckets, len(breaches))
    size = len(breaches) / buckets
    out: list[tuple[int, int, float]] = []
    for index in range(buckets):
        start = int(round(index * size))
        end = max(int(round((index + 1) * size)), start + 1)
        out.append((start + 1, end, pooled(breaches[start:end], applicable[start:end])))
    return out





@dataclass(frozen=True)
class StoredLesson:
    title: str
    query: str
    op_id: str | None
    found: bool
    impressions: int = 0


@dataclass(frozen=True)
class Reconciliation:
    """What the run submitted, against what the store actually holds.

    An acknowledged write is not a permanent one. Duplicates are consolidated
    into endorsements of an existing entry, and the quality gate can decline
    something after the fact, so the count of `create_memory` calls that
    returned an operation id says what was offered, not what was kept. Only the
    store can say that, which is why this asks it.

    Two things this deliberately does not claim. Matching is by title, using
    each lesson's own query, so `absent` is a floor rather than a fact: a lesson
    the server kept under a rewritten title looks absent here. And consolidation
    is not counted, because nothing observable distinguishes it: the store's
    `impressions` figure rises on retrieval as well as on a duplicate write, so
    counting merges with it would count this reconciliation's own probes. What
    an absent title means is settled by asking the store the question that
    lesson answers, which is a judgement for the reader, not a number.
    """

    lessons: tuple[StoredLesson, ...] = ()
    probes: int = 0

    @property
    def submitted(self) -> int:
        return len(self.lessons)

    @property
    def stored(self) -> int:
        return sum(1 for lesson in self.lessons if lesson.found)

    @property
    def absent(self) -> tuple[StoredLesson, ...]:
        return tuple(lesson for lesson in self.lessons if not lesson.found)


def reconcile(
    run: Run,
    search: Callable[[str], Any],
    pace: Callable[[float], None] | None = None,
    pace_seconds: float = 6.0,
) -> Reconciliation:
    """Probe the store for everything the run wrote. Read-only.

    `search` is injected rather than a client built here, so this stays testable
    offline and so the caller owns the credentials. Nothing in this path writes
    or sends feedback: reconciliation must not change what it is measuring.

    Probes are paced for the same reason episodes are, and identical queries are
    asked once, because several lessons about the same thing tend to share one.
    """
    written = [
        lesson
        for record in run.records
        for lesson in record.get("lessons_written", [])
        if lesson.get("title")
    ]
    by_query: dict[str, list[dict[str, Any]]] = {}
    for lesson in written:
        by_query.setdefault(lesson.get("query") or lesson["title"], []).append(lesson)

    seen: dict[str, int] = {}
    probes = 0
    for index, query in enumerate(by_query):
        if index and pace is not None and pace_seconds > 0:
            pace(pace_seconds)
        result = search(query)
        probes += 1
        for memory in getattr(result, "memories", ()) or ():
            for insight in memory.insights:
                seen[_key(insight.title)] = max(
                    seen.get(_key(insight.title), 0), memory.impressions
                )

    lessons = tuple(
        StoredLesson(
            title=lesson["title"],
            query=lesson.get("query", ""),
            op_id=lesson.get("op_id"),
            found=_key(lesson["title"]) in seen,
            impressions=seen.get(_key(lesson["title"]), 0),
        )
        for lesson in written
    )
    return Reconciliation(lessons=lessons, probes=probes)


def _key(title: str) -> str:
    return " ".join(title.lower().split())


def render_reconciliation(result: Reconciliation) -> str:
    lines = [
        "reconciliation: what the store holds, against what the run submitted",
        f"  submitted {result.submitted}, found under their own title {result.stored}, "
        f"absent {len(result.absent)}  ({result.probes} read-only probes)",
    ]
    if result.absent:
        lines.append("  titles the store did not return:")
        for lesson in result.absent:
            lines.append(f"    {lesson.title}   (op {lesson.op_id})")
        lines.append(
            "  absent means the title did not come back under its own query. That is "
            "consolidation into an entry written earlier, a title the server rewrote, "
            "or a declined write, and the three look identical from here: ask the store "
            "the question each lesson answers to tell them apart."
        )
    return "\n".join(lines)



def grader_warning(runs: list[Run]) -> str | None:
    versions: set[str] = set()
    for run in runs:
        versions |= run.grader_versions
    versions.discard("?")
    if len(versions) > 1:
        return (
            f"results mix grader versions ({', '.join(sorted(versions))}); "
            "they were not scored under the same rules and should not be compared"
        )
    if versions and versions != {GRADER_VERSION}:
        return (
            f"results were scored under grader version {versions.pop()}, "
            f"this build is version {GRADER_VERSION}"
        )
    return None


# --- terminal -----------------------------------------------------------------


def render_report(run: Run, control: Run | None = None) -> str:
    """The terminal summary: the headline comparison and how it moved."""
    lines: list[str] = []
    abandoned = run.meta.get("abandoned")
    if abandoned:
        lines.append(
            f"THIS RUN IS NOT A MEASUREMENT. It was abandoned after "
            f"{abandoned['after_pair']} task(s) because "
            f"{abandoned['consecutive_failures']} in a row hit errors "
            f"({abandoned['reason']}). It therefore stopped being a comparison "
            "part way through. Read the figures below as diagnosis, never as a result."
        )
        lines.append("")
    warning = grader_warning([run, control] if control else [run])
    if warning:
        lines.append(f"warning: {warning}")
        lines.append("")

    heading = pairing_of(run)
    if heading:
        lines.append(preamble(heading.get("pairs", len(run.records))))
        lines.append("")
    selection = "all variants" if run.meta.get("all_variants") else f"seed {run.seed}"
    lines.append(f"run {run.run_id}")
    lines.append(
        f"  {selection}, {len(run.records)} episodes, "
        f"agent {run.meta.get('agent_model', '?')}, "
        f"reviewer {run.meta.get('reviewer_model', '?')}"
    )
    lines.append(
        f"  compliance: {run.compliance:.0f}% of applicable policies met "
        f"({sum(run.applicable) - sum(run.breaches)} of {sum(run.applicable)}), "
        f"{run.mean:.2f} breaches per episode"
    )
    usage = run.meta.get("usage") or {}
    if usage:
        lines.append(
            f"  tokens: {usage.get('input_tokens', 0):,} in, "
            f"{usage.get('output_tokens', 0):,} out, over {usage.get('calls', 0)} model calls"
        )
    if control:
        lines.append(f"control {control.run_id} (no memory)")
        lines.append(
            f"  compliance: {control.compliance:.0f}% of applicable policies met "
            f"({sum(control.applicable) - sum(control.breaches)} of "
            f"{sum(control.applicable)}), {control.mean:.2f} breaches per episode"
        )
    pairing = pairing_of(run)
    if pairing:
        lines.append("")
        lines.append(f"how the two arms compared over {pairing.get('pairs', 0)} task(s)")
        lines.append(f"  without memory  {pairing.get('control_compliance', 0):.0f}% "
                     "of applicable policies met")
        lines.append(f"  with memory     {pairing.get('memory_compliance', 0):.0f}% "
                     "of applicable policies met")
        endpoints = pairing.get("confidence_interval")
        if endpoints:
            half = (endpoints[1] - endpoints[0]) / 2
            lines.append(f"  memory advantage {pairing.get('mean_difference', 0):.0f} "
                         f"±{half:.0f} percentage points")
        lines.append(
            f"  underneath: {pairing.get('control_mean', 0):.2f} rule breaches per "
            f"task without memory, {pairing.get('memory_mean', 0):.2f} with"
        )
        dropped_pairs = int(pairing.get("excluded_pairs") or 0)
        if dropped_pairs:
            lines.append(
                f"  {dropped_pairs} pair(s) excluded: review failed. Both halves were "
                "answered, neither was scored, and a pair scored on one arm only is "
                "not a comparison"
            )
        drift = pairing.get("control_drift")
        if drift is not None and abs(drift) >= 2:
            lines.append(f"  {drift_note(drift)}")
        if run.meta.get("interrupted"):
            lines.append("  this run was stopped early; the figures cover the tasks finished")
        lines.append(
            "  a run can be stopped at any point and this still reads; the figures "
            "quoted in this repository's documentation come from complete runs of a "
            "fixed length."
        )
    failed = sum(1 for record in run.records if record.get("review_failed"))
    if failed:
        lines.append(f"  {failed} episode(s) recorded as review_failed and left out of the table")
    lines.append("")

    lines.append("compliance (%), by decile")
    header = f"  episodes  {run.arm:>10}"
    if control:
        header += "   control"
    lines.append(header)
    control_buckets = deciles(control.breaches, control.applicable) if control else []
    for index, (start, end, met) in enumerate(deciles(run.breaches, run.applicable)):
        bucket = f"  {start:>4}-{end:<5}{met:>10.0f}"
        if index < len(control_buckets):
            bucket += f"     {control_buckets[index][2]:>5.0f}"
        lines.append(bucket)
    lines.append("")

    return "\n".join(lines)


# --- HTML ---------------------------------------------------------------------


def write_html(run: Run, control: Run | None, out_path: Path, live: bool = False) -> Path:
    """Render the report. `live` marks a run still in progress.

    A live page refreshes itself; a finished one does not, so the file left
    behind is a self-contained artefact that can be sent to somebody as one
    HTML file rather than a page that keeps trying to reload.
    """
    out_path.write_text(_html(run, control, live=live), encoding="utf-8")
    return out_path


def _html(run: Run, control: Run | None, live: bool = False) -> str:
    warning = grader_warning([run, control] if control else [run])
    warning_block = (
        f'<p class="warn">Warning: {html.escape(warning)}</p>' if warning else ""
    )
    selection = "all variants" if run.meta.get("all_variants") else f"seed {run.seed}"

    # A memory-off run has no learning curve to show: the same page has to
    # describe a baseline honestly rather than promise a decline that the arm
    # was never able to produce.
    if run.memory_enabled:
        curve_heading = "The learning curve"
        curve_blurb = (
            "Compliance is the share of the policies that applied to a task that the "
            f"reply got right; higher is better. Each point pools the last {WINDOW} "
            "tasks: obligations met over obligations that applied across the window. "
            "The memory line climbs as lessons accumulate, while the arm answering "
            "the same emails without memory stays put."
        )
        advantage_blurb = (
            "The third line is the memory advantage: how far ahead of the blind arm "
            "memory is on the same task, in percentage points, averaged over every "
            "task so far. Above the zero line means memory got more of the policies "
            "right. Its band is how much that average could still move, and starts "
            f"at task {ADVANTAGE_FLOOR}; before that there are too few tasks to put "
            "a band on."
        )
    else:
        curve_heading = "Compliance"
        curve_blurb = (
            "Compliance is the share of the policies that applied to a task that the "
            f"reply got right; higher is better. Each point pools the last {WINDOW} "
            "tasks. Memory was off for this run, so nothing accumulated between tasks "
            "and the line is a baseline rather than a learning curve."
        )
        advantage_blurb = ""

    # A live page reloads itself; a finished one is a file somebody can send on.
    refresh_tag = '<meta http-equiv="refresh" content="5">' if live else ""
    abandoned = run.meta.get("abandoned")
    abandoned_block = (
        '<p class="warn"><strong>This run is not a measurement.</strong> It was '
        f"abandoned after {abandoned['after_pair']} task(s) because "
        f"{abandoned['consecutive_failures']} in a row hit errors "
        f"({html.escape(str(abandoned['reason']))}). It therefore stopped being a "
        "comparison part way through. Read what follows as diagnosis, not as a "
        "result.</p>"
        if abandoned
        else ""
    )
    live_block = (
        '<p class="live">This run is still going. The page reloads itself every '
        "five seconds.</p>"
        if live
        else ""
    )
    pairing = pairing_of(run)
    preamble_block = ""
    # The comparison used to have a chart of its own. It is the third line on the
    # curve now, so what is left of that section is the reading of it: what the
    # advantage came to, when it became reliable, and whether the sample moved
    # underneath it.
    comparison_block = ""
    if pairing:
        preamble_block = (
            '<p class="preamble">'
            + "<br>".join(html.escape(line) for line in
                          preamble(pairing.get("pairs", len(run.records))).splitlines())
            + "</p>"
        )
        headline = (
            f'<p class="headline">{pairing.get("memory_compliance", 0):.0f}% with '
            f'memory against {pairing.get("control_compliance", 0):.0f}% without, '
            f'over {pairing.get("pairs", 0)} tasks answered twice.</p>'
        )
        endpoints = pairing.get("confidence_interval")
        working = (
            f'<p class="prov">Memory advantage over the run as a whole: '
            f'{pairing.get("mean_difference", 0):.1f} percentage points ahead of the '
            f'arm answering the same email without memory, somewhere between '
            f'{endpoints[0]:.1f} and {endpoints[1]:.1f} (95% confidence interval on '
            f'the paired differences). Underneath the percentages: '
            f'{pairing.get("control_mean", 0):.2f} rule breaches per task without '
            f'memory, {pairing.get("memory_mean", 0):.2f} with.</p>'
            if endpoints else ""
        )
        dropped_pairs = int(pairing.get("excluded_pairs") or 0)
        excluded_note = (
            f'<p class="note">{dropped_pairs} pair(s) excluded: review failed. Both '
            "halves were answered, neither was scored, and a pair scored on one arm "
            "only is not a comparison.</p>"
            if dropped_pairs
            else ""
        )
        drift = pairing.get("control_drift")
        drift_note = ""
        if drift is not None and abs(drift) >= 2:
            drift_note = f'<p class="note">{html.escape(_drift(drift))}</p>'
        comparison_block = f"""{headline}
{working}
{excluded_note}
{drift_note}
<p class="note">A run can be stopped at any point and this page still reads; the
figures quoted in this repository's documentation come from complete runs of a
fixed length.</p>"""

    control_row = (
        f"<tr><th>Without memory</th><td>{control.run_id}</td>"
        f"<td>{control.compliance:.0f}%</td><td>{control.mean:.2f}</td>"
        f"<td>{len(control.records)}</td></tr>"
        if control
        else ""
    )

    # Named from the run id rather than from `path`, which is the page's own
    # location when the runner renders live and the JSONL's when the reporter
    # does. The arm suffix comes off: both arms live in the one file.
    stem = run.run_id.split(":")[0]
    records_file = f"results/{stem}.jsonl"
    pairs_file = f"results/{stem}.pairs.jsonl"

    return f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_tag}
<title>Learning on the job — {html.escape(run.run_id)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0 auto; max-width: 52rem; padding: 2.5rem 1.25rem 4rem; }}
  h1 {{ font-size: 1.5rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2.5rem; }}
  p.sub {{ color: #6b7280; margin-top: 0; }}
  p.warn {{ background: #fef3c7; color: #7c2d12; padding: .6rem .8rem; border-radius: 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
  th, td {{ text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #e5e7eb; }}
  th {{ font-weight: 600; }}
  code {{ font-size: .88em; }}
  .prov, .note {{ color: #6b7280; font-size: .85rem; }}
  .legend span {{ margin-right: 1rem; font-size: .85rem; }}
  p.preamble {{ background: #f3f4f6; border-left: 3px solid #9ca3af;
                padding: .7rem .9rem; border-radius: 4px; }}
  p.live {{ background: #dbeafe; color: #1e3a8a; padding: .5rem .8rem;
            border-radius: 6px; font-size: .9rem; }}
  p.headline {{ font-size: 1.05rem; font-weight: 600; }}
  .swatch {{ display: inline-block; width: 1.1rem; height: .2rem; vertical-align: middle;
             margin-right: .35rem; }}
  /* The advantage line is dashed on the chart, so its swatch is too: the
     legend has to look like the thing it labels. */
  .swatch.dashed {{ background: repeating-linear-gradient(90deg,
      {ADVANTAGE_COLOUR} 0 .35rem, transparent .35rem .55rem); }}
  @media (prefers-color-scheme: dark) {{
    th, td {{ border-bottom-color: #374151; }}
    p.warn {{ background: #422006; color: #fde68a; }}
  }}
</style>
</head>
<body>
<h1>Learning on the job</h1>
<p class="sub">{html.escape(run.run_id)} &middot; {html.escape(selection)}
&middot; {len(run.records)} episodes &middot; agent
{html.escape(str(run.meta.get('agent_model', 'unknown')))}</p>
{abandoned_block}
{live_block}
{preamble_block}
{warning_block}

<h2>{html.escape(curve_heading)}</h2>
<p>{html.escape(curve_blurb)}</p>
{f'<p>{html.escape(advantage_blurb)}</p>' if advantage_blurb and control else ''}
{_svg(run, control)}
<p class="legend"><span><i class="swatch"
 style="background:{MEM_COLOUR}"></i>{html.escape(run.arm)}</span>
{f'<span><i class="swatch" style="background:{NO_MEM_COLOUR}"></i>no memory</span>'
 if control else ''}
{'<span><i class="swatch dashed"></i>memory advantage</span>' if control else ''}</p>
{comparison_block}

<table>
<tr><th>Arm</th><th>Run</th><th>Compliance</th><th>Breaches / episode</th>
<th>Episodes</th></tr>
<tr><th>{html.escape(run.arm.capitalize())}</th><td>{html.escape(run.run_id)}</td>
<td>{run.compliance:.0f}%</td><td>{run.mean:.2f}</td><td>{len(run.records)}</td></tr>
{control_row}
</table>

<p class="note">This page is the result. Everything behind it is recorded per task
in <code>{html.escape(records_file)}</code>, one line per episode: what memory
returned, the feedback sent back on it, the lessons written, which policies each
reply breached, and whether memory had already said so. Pair-level figures are in
<code>{html.escape(pairs_file)}</code>. Both are the place to start for your own
analysis, on your own scenario.</p>
</body>
</html>
"""


# The three lines. Blue and amber are a checked pair: colour-blind separation
# and contrast against the page both clear their thresholds, so the two lines a
# reader is asked to tell apart do not rely on hue alone being kind. The
# no-memory arm stays a neutral grey on purpose. It is the baseline the other
# two are read against rather than a third thing competing for attention, and
# the legend and the dashing carry its identity where the colour does not.
MEM_COLOUR = "#2563eb"
NO_MEM_COLOUR = "#6b7280"
ADVANTAGE_COLOUR = "#d97706"

# Gridline steps a reader can do arithmetic in. Sizing them off the data gave
# axes labelled 1.1, 2.2, 3.3, which are the right lines in the wrong places.
_NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0)


def memory_advantage(run: Run, control: Run | None) -> list[tuple[float, float | None]]:
    """How far ahead memory is on the same task, averaged over the run so far.

    The same quantity the terminal prints under `memory advantage`: the mean of
    the per-task compliance differences, in percentage points, with the interval
    around it. Pairing is what makes it worth drawing: both arms answered the
    same email, so the task's own difficulty cancels and what is left is the
    memory.

    Per-task differences, not a difference of the two pooled lines. The pair is
    the unit the interval is computed over, and it is only because each task
    contributes one number that the task-to-task variation cancels at all.
    """
    if control is None:
        return []
    pairs = min(len(run.breaches), len(control.breaches))
    if pairs < 2:
        return []
    applicable = run.applicable
    return running_intervals([
        compliance(run.breaches[index], applicable[index])
        - compliance(control.breaches[index], applicable[index])
        for index in range(pairs)
    ])


def _svg(run: Run, control: Run | None) -> str:
    """The one chart: both arms' compliance, and how far apart they are.

    Three lines on a single axis, all of them percentages of the same thing. The
    two arms are compliance pooled over the trailing window, which is the shape
    a reader follows; the memory advantage is the running average of the paired
    differences in percentage points, drawn as its own visual family rather than
    as a third arm, with the band that says how much it could still move.

    The axis runs to 100 whatever the data does. A chart scaled to the tallest
    line would redraw its own axis every ten tasks, and a climb towards the top
    of the frame would mean nothing without knowing where the top was.
    """
    width, height = 720, 290
    pad_left, pad_right, pad_top, pad_bottom = 46, 12, 16, 34
    series = rolling_compliance(run.breaches, run.applicable)
    control_series = (
        rolling_compliance(control.breaches, control.applicable) if control else []
    )
    if not series and not control_series:
        return "<p>No tasks to plot.</p>"
    advantage = memory_advantage(run, control)

    count = max(len(series), len(control_series), 2)
    lows = [mean - (half or 0.0) for mean, half in advantage]
    top = 100.0
    # Zero has to be on the chart whatever the data does: the whole reading of
    # the advantage line is which side of it the line sits. It drops below only
    # far enough to keep a negative band inside the frame.
    bottom = min([0.0, *lows]) * 1.15
    span = top - bottom or 1.0
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def coords(index: int, value: float) -> tuple[float, float]:
        x = pad_left + (plot_w * index / max(1, count - 1))
        y = pad_top + plot_h - (plot_h * (value - bottom) / span)
        return x, y

    def line(
        points: list[tuple[int, float]], colour: str, thickness: float = 2, dash: str = ""
    ) -> str:
        if len(points) < 2:
            return ""
        drawn = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                         (coords(index, value) for index, value in points))
        style = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline fill="none" stroke="{colour}" stroke-width="{thickness}" '
            f'stroke-linejoin="round"{style} points="{drawn}" />'
        )

    def arm(values: list[float], colour: str) -> str:
        return line(list(enumerate(values)), colour)

    def band() -> str:
        """The interval around the advantage, once there is enough to draw one."""
        banded = [(i, m, h) for i, (m, h) in enumerate(advantage) if h is not None]
        if len(banded) < 2:
            return ""
        upper = [coords(i, m + h) for i, m, h in banded]
        lower = [coords(i, m - h) for i, m, h in reversed(banded)]
        drawn = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        return f'<polygon points="{drawn}" fill="{ADVANTAGE_COLOUR}" fill-opacity="0.14" />'

    def advantage_lines() -> str:
        """Thin until the interval exists, full weight once it does.

        The change of weight is the same convention as the em-dash the terminal
        prints for its first nine pairs: the number is there from the first
        pair, and drawing it at full weight before it can be qualified would
        offer a precision it does not yet have.
        """
        if not advantage:
            return ""
        means = [(index, mean) for index, (mean, _) in enumerate(advantage)]
        # The two segments share the pair where the band starts, so they join.
        return line(means[:ADVANTAGE_FLOOR], ADVANTAGE_COLOUR, 1, "3 3") + line(
            means[ADVANTAGE_FLOOR - 1 :], ADVANTAGE_COLOUR, 2, "6 4"
        )

    def gridlines() -> str:
        step = next((s for s in _NICE_STEPS if s >= span / 5), _NICE_STEPS[-1])
        drawn = []
        for tick in range(math.ceil(bottom / step), math.floor(top / step) + 1):
            value = tick * step
            _, y = coords(0, value)
            # Zero is the line the advantage is read against, so it is darker
            # than the rest of the grid and never just another gridline.
            colour = "#9ca3af" if value == 0 else "#e5e7eb"
            drawn.append(
                f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
                f'stroke="{colour}" stroke-width="1" />'
                f'<text x="{pad_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="#6b7280">{value:g}</text>'
            )
        return "".join(drawn)

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img"
 aria-label="Compliance for both arms pooled over the last {WINDOW} tasks, and how
 far ahead memory is in percentage points with its 95% band">
{gridlines()}
{band()}
{arm(control_series, NO_MEM_COLOUR)}
{arm(series, MEM_COLOUR)}
{advantage_lines()}
<text x="{pad_left}" y="{height - 10}" font-size="11" fill="#6b7280">task 1</text>
<text x="{width - pad_right}" y="{height - 10}" font-size="11" fill="#6b7280"
 text-anchor="end">task {count}</text>
</svg>"""
