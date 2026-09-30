"""The episode loop.

Pick the episodes, run each one through agent → reviewer → record → reflection,
and append the result as it completes so an interrupted run keeps its data.

Two ways to pick episodes. Normally the seed samples `task_count` tasks without
replacement and chooses one variant each; neither depends on whether memory is
on, so the control arm sees exactly the same emails in the same order, which is
what makes the two curves comparable. With `--all-variants` the run instead
walks the whole library in file order and runs every variant of every task, a
fixed list with no sampling — used to measure the scenario itself rather than a
learning curve.

With `--paired`, one loop answers each task twice: once by an agent that can
search memory and writes back afterwards, then once by an agent that is blind to
memory entirely. Both arms get the identical email. This replaces running the
two arms as separate invocations, and it is a better measurement rather than
merely a tidier one: tasks differ enormously in how much there is to get wrong,
and comparing two separate runs compares two draws from that variation. Pairing
cancels the task out, so what is left is the difference memory made. The control
episode also occupies the gap that has to be left between memory episodes for a
write to become searchable, so the arm costs wall-clock that was being spent on
sleeping anyway.

Nothing here reads the wall clock for scenario purposes: the agent and the
reviewer are both given `orders.yaml`'s `reference_date` as today.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from . import display
from .agent import draft_reply
from .grader_version import GRADER_VERSION
from .memco_client import MemcoClient, _describe, build_client, stamp
from .pairing import PairStats
from .providers import Provider, Usage, build_provider, model_for
from .reflection import Reflection, Reflector
from .reviewer import Reviewer
from .scenario import Scenario, Task, load_scenario

__all__ = ["RunConfig", "RunResult", "run"]

DEFAULT_RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class RunConfig:
    task_count: int = 30
    seed: int = 42
    memory_enabled: bool = True
    all_variants: bool = False
    # One loop, both arms, the same email to each. See `_run_paired`.
    paired: bool = False
    # Stop once this many consecutive tasks hit errors. A single failure is a
    # hiccup and the loop carries on; a run of them means the memory arm is
    # drafting blind, and every further task spends money producing a comparison
    # that is no longer a comparison. Zero turns the check off.
    error_streak_limit: int = 3
    # Continue a paired run that stopped part way, by its run id. The same seed
    # gives the same tasks in the same order, and the lessons the first attempt
    # wrote are still in the store, so picking up where it left off is the run
    # continuing rather than a second run pretending to be the first. Only valid
    # if the store has not been cleared in between.
    resume_from: str = ""
    label: str = ""  # goes in the run id, e.g. "pass1", to tell repeats apart
    pace_seconds: float = 10.0  # gap held between episodes; see `_pace`
    results_dir: Path = DEFAULT_RESULTS_DIR
    scenario_root: Path | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    jsonl_path: Path
    meta_path: Path
    records: tuple[dict[str, Any], ...] = ()


@dataclass
class _Providers:
    agent: Provider
    reviewer: Provider
    reflection: Provider | None = None

    def usage(self) -> Usage:
        total = self.agent.usage + self.reviewer.usage
        return total + self.reflection.usage if self.reflection else total

    def usage_by_role(self) -> dict[str, dict[str, int]]:
        """Tokens per role, so a run whose roles sit on different providers can be priced."""
        roles = {"agent": self.agent, "reviewer": self.reviewer}
        if self.reflection:
            roles["reflection"] = self.reflection
        # One provider object can serve two roles in a test or a single-model
        # run; splitting a shared total between them would invent numbers.
        if len({id(provider) for provider in roles.values()}) != len(roles):
            return {}
        return {role: provider.usage.as_record() for role, provider in roles.items()}

    def as_record(self) -> dict[str, str]:
        record = {"agent_model": self.agent.spec, "reviewer_model": self.reviewer.spec}
        if self.reflection:
            record["reflection_model"] = self.reflection.spec
        return record

    def adaptations(self) -> dict[str, list[str]]:
        """Compatibility adaptations an endpoint turned out to need, by role.

        These change how the model works, so they belong in the run's metadata
        next to the model names rather than being swallowed by the adapter.
        """
        roles = (("agent", self.agent), ("reviewer", self.reviewer))
        if self.reflection:
            roles += (("reflection", self.reflection),)
        found = {
            role: sorted(getattr(provider, "adaptations", None) or ())
            for role, provider in roles
        }
        return {role: names for role, names in found.items() if names}


def run(
    config: RunConfig,
    scenario: Scenario | None = None,
    agent_provider: Provider | None = None,
    reviewer_provider: Provider | None = None,
    reflection_provider: Provider | None = None,
    memory: MemcoClient | None = None,
    report: Callable[[str], None] = print,
) -> RunResult:
    """Run the selected episodes and write the results.

    Everything the run depends on can be passed in, which is how the tests run
    the whole loop offline.

    A client built here is also closed here, so the connection lasts exactly as
    long as the run that needs it. One passed in belongs to the caller and is
    left alone.
    """
    owned: MemcoClient | None = None
    if config.memory_enabled and memory is None:
        memory = owned = build_client()
    try:
        return _run_episodes(
            config,
            scenario=scenario,
            agent_provider=agent_provider,
            reviewer_provider=reviewer_provider,
            reflection_provider=reflection_provider,
            memory=memory,
            report=report,
        )
    finally:
        if owned is not None:
            owned.close()


def _run_episodes(
    config: RunConfig,
    scenario: Scenario | None = None,
    agent_provider: Provider | None = None,
    reviewer_provider: Provider | None = None,
    reflection_provider: Provider | None = None,
    memory: MemcoClient | None = None,
    report: Callable[[str], None] = print,
) -> RunResult:
    scenario = scenario or load_scenario(config.scenario_root)
    if not config.memory_enabled:
        memory = None

    # Built per role, and only for the roles this run uses: with memory off
    # there is no reflection step, so its provider is never constructed and its
    # credentials are never needed.
    providers = _Providers(
        agent=agent_provider or build_provider(model_for("agent")),
        reviewer=reviewer_provider or build_provider(model_for("reviewer")),
    )
    if memory is not None:
        providers.reflection = reflection_provider or build_provider(model_for("reflection"))

    reviewer = Reviewer(
        provider=providers.reviewer,
        policies=scenario.policies,
        reference_date=scenario.reference_date,
    )
    reflector = (
        Reflector(provider=providers.reflection)
        if memory is not None and providers.reflection is not None
        else None
    )

    started = datetime.now(UTC)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    if config.resume_from:
        run_id = config.resume_from
        jsonl_path = config.results_dir / f"{run_id}.jsonl"
        if not jsonl_path.exists():
            raise RuntimeError(f"no run to resume at {jsonl_path}")
        prior = _complete_pairs(jsonl_path)
    else:
        run_id = _unique_run_id(_run_id(started, config), config.results_dir)
        jsonl_path = config.results_dir / f"{run_id}.jsonl"
        prior = []
    meta_path = config.results_dir / f"{run_id}.meta.json"

    selection = _select(scenario, config)
    _write_meta(meta_path, run_id, config, providers, scenario, started, len(selection))
    paced = config.pace_seconds > 0 and memory is not None
    if config.paired:
        report(display.preamble(len(selection)))
        report(display.configuration(
            None if config.all_variants else config.seed,
            providers.agent.spec,
            providers.reviewer.spec,
        ))
        report("")
        records, stats, interrupted, abandoned, excluded_review = _run_paired(
            config, scenario, selection, providers, reviewer, reflector, memory,
            run_id, jsonl_path, report, prior,
        )
        _write_meta(
            meta_path, run_id, config, providers, scenario, started, len(selection),
            finished=datetime.now(UTC),
            total_breaches=sum(stats.memory_breaches),
            pairing={**stats.as_record(), "excluded_pairs": excluded_review},
            interrupted=interrupted,
            abandoned=abandoned,
        )
        _report_pairs(stats, interrupted, report, abandoned, excluded_review)
        return RunResult(
            run_id=run_id, jsonl_path=jsonl_path, meta_path=meta_path,
            records=tuple(records),
        )

    report(
        f"{run_id}: {len(selection)} episodes, memory {'on' if memory else 'off'}, "
        f"today is {scenario.reference_date}"
        + (f", pacing {config.pace_seconds:g}s between episodes" if paced else "")
    )

    records: list[dict[str, Any]] = []
    breaches_so_far = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, (task, variant_index) in enumerate(selection):
            if index:
                _pace(config.pace_seconds, memory is not None, sleep)
            before = providers.usage()
            record = _run_task(
                index=index,
                run_id=run_id,
                task=task,
                variant_index=variant_index,
                scenario=scenario,
                providers=providers,
                reviewer=reviewer,
                reflector=reflector,
                memory=memory,
            )
            after = providers.usage()
            record["usage"] = {
                "input_tokens": after.input_tokens - before.input_tokens,
                "output_tokens": after.output_tokens - before.output_tokens,
                "calls": after.calls - before.calls,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            records.append(record)
            breaches_so_far += record["breach_count"]
            report(
                f"[{index + 1:>3}/{len(selection)}] {task.id}/v{variant_index}  "
                f"breaches={record['breach_count']}  "
                f"lessons={len(record['lessons_written'])}  "
                f"mean={breaches_so_far / (index + 1):.2f}"
                + ("  REVIEW FAILED" if record["review_failed"] else "")
            )

    _write_meta(
        meta_path,
        run_id,
        config,
        providers,
        scenario,
        started,
        len(selection),
        finished=datetime.now(UTC),
        total_breaches=breaches_so_far,
    )
    return RunResult(
        run_id=run_id,
        jsonl_path=jsonl_path,
        meta_path=meta_path,
        records=tuple(records),
    )


def _run_paired(
    config: RunConfig,
    scenario: Scenario,
    selection: list[tuple[Task, int]],
    providers: _Providers,
    reviewer: Reviewer,
    reflector: Reflector | None,
    memory: MemcoClient | None,
    run_id: str,
    jsonl_path: Path,
    report: Callable[[str], None],
    prior: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], PairStats, bool, dict[str, Any] | None, int]:
    """Answer every task twice, once with memory and once without.

    The two arms see the same email, in the same order, from the same seed, and
    differ in one thing only: whether the agent could search memory and whether
    anything was written back afterwards. Running them interleaved rather than
    as two separate runs means the comparison is within a task rather than
    across two draws from a library whose tasks vary enormously in how much
    there is to get wrong.
    """
    records: list[dict[str, Any]] = list(prior or [])
    stats = PairStats()
    # Replay what the earlier attempt measured, so the trailing means, the
    # trailing compliance and the interval all continue rather than starting
    # again from a standing start halfway through a run.
    # A set, not a watermark: a run can leave a hole in the middle when one task
    # failed and the ones after it succeeded, and those holes have to be filled
    # rather than jumped over.
    measured: set[int] = set()
    if records:
        by_index: dict[int, dict[str, dict[str, Any]]] = {}
        for record in records:
            by_index.setdefault(record["pair_index"], {})[record["arm"]] = record
        for index in sorted(by_index):
            both = by_index[index]
            if "memory" in both and "control" in both:
                stats.add(
                    both["memory"]["breach_count"],
                    both["control"]["breach_count"],
                    len(both["memory"]["applicable_policies"]),
                )
                measured.add(index)
    done = len(measured)
    pairs_path = jsonl_path.with_suffix(".pairs.jsonl")
    html_path = jsonl_path.with_suffix(".report.html")
    interrupted = False
    abandoned: dict[str, Any] | None = None
    failing_streak = 0
    excluded_review = 0
    report(display.header())

    def episode(index: int, task: Task, variant_index: int, arm: str) -> dict[str, Any]:
        before = providers.usage()
        try:
            record = _episode(index, task, variant_index, arm)
        except Exception as error:  # noqa: BLE001 - one episode is not the run
            # A model call can time out or a provider can refuse, and losing an
            # hour of paid work to one of them is not acceptable. A failed
            # episode is recorded as failed rather than as a good episode with
            # no breaches, because a fabricated zero would flatter whichever arm
            # it landed on. The pair is dropped from the comparison and the
            # streak counter picks it up if this keeps happening.
            record = {
                "run_id": run_id,
                "task_index": index,
                "task_id": task.id,
                "variant_index": variant_index,
                "arm": arm,
                "pair_index": index,
                "memory_enabled": arm == "memory",
                "applicable_policies": list(task.applicable_policies),
                "breaches": [],
                "breach_count": 0,
                "episode_failed": True,
                "review_failed": False,
                "dropped_policies": [],
                "lessons_written": [],
                "lessons_retrieved": [],
                "feedback_sent": [],
                "feedback_calls": [],
                "draft": "",
                "corrected_draft": "",
                "errors": [f"{arm} episode failed: {_describe(error)}"],
                **providers.as_record(),
                "grader_version": GRADER_VERSION,
                "at": stamp(),
            }
        after = providers.usage()
        record["usage"] = {
            "input_tokens": after.input_tokens - before.input_tokens,
            "output_tokens": after.output_tokens - before.output_tokens,
            "calls": after.calls - before.calls,
        }
        return record

    def _episode(index: int, task: Task, variant_index: int, arm: str) -> dict[str, Any]:
        record = _run_task(
            index=index,
            run_id=run_id,
            task=task,
            variant_index=variant_index,
            scenario=scenario,
            providers=providers,
            reviewer=reviewer,
            # The control arm is blind, not merely unaided: no search while it
            # drafts, and nothing written, graded or fed back afterwards.
            reflector=reflector if arm == "memory" else None,
            memory=memory if arm == "memory" else None,
            arm=arm,
            pair_index=index,
        )
        return record

    if done:
        report(f"resuming after {done} completed task(s)")
        # Any half-written pair from the crash is dropped: the file is rewritten
        # to the pairs that finished, so a resumed episode never lands beside
        # the orphaned half of the one that did not.
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    mode = "a" if done else "w"
    with jsonl_path.open(mode, encoding="utf-8") as handle, \
            pairs_path.open(mode, encoding="utf-8") as pairs_handle:

        def emit(target: Any, record: dict[str, Any]) -> None:
            target.write(json.dumps(record) + "\n")
            target.flush()

        try:
            for index, (task, variant_index) in enumerate(selection):
                if index in measured:
                    continue  # already measured by the attempt being resumed
                with_memory = episode(index, task, variant_index, "memory")
                emit(handle, with_memory)
                records.append(with_memory)

                # Pacing is measured from here. The control episode is real work
                # that takes real time, and that time is exactly the indexing
                # allowance the gap exists to provide, so it counts towards the
                # gap rather than being added to it.
                since_memory = monotonic()
                control = episode(index, task, variant_index, "control")
                emit(handle, control)
                records.append(control)

                # A pair measures something only if both halves were scored. An
                # episode that fell over and a review that could not be read both
                # leave a half with no score, and the zero each was recorded with
                # is the absence of a verdict rather than a faultless reply. The
                # report has always left them out; the loop used to leave out
                # only the first kind, so a failed review reached the live table,
                # the pairs file and the interval as a clean sweep for whichever
                # arm it landed on.
                broke = with_memory.get("episode_failed") or control.get("episode_failed")
                unscored = with_memory["review_failed"] or control["review_failed"]
                if broke or unscored:
                    for problem in sorted({*with_memory["errors"], *control["errors"]}):
                        report(display.note(problem))
                    if broke:
                        report(display.note(
                            f"task {index + 1} dropped from the comparison: an episode did "
                            "not complete, and a pair needs both halves"
                        ))
                        failing_streak += 1
                        if (config.error_streak_limit
                                and failing_streak >= config.error_streak_limit):
                            abandoned = {
                                "after_pair": index + 1,
                                "consecutive_failures": failing_streak,
                                "reason": (with_memory["errors"] or control["errors"])[0],
                            }
                            report(display.abandoned(failing_streak, abandoned["reason"]))
                            break
                    else:
                        excluded_review += 1
                        report(display.note(
                            f"task {index + 1} excluded from the comparison: the reviewer's "
                            "answer could not be read, so one half of the pair has no score"
                        ))
                    if index + 1 < len(selection):
                        _pace_floor(config.pace_seconds, monotonic() - since_memory, sleep)
                    continue

                applicable = len(task.applicable_policies)
                stats.add(with_memory["breach_count"], control["breach_count"], applicable)
                emit(pairs_handle, {
                    "run_id": run_id,
                    "pair_index": index,
                    "task_id": task.id,
                    "variant_index": variant_index,
                    "applicable": applicable,
                    "memory_breaches": with_memory["breach_count"],
                    "control_breaches": control["breach_count"],
                    "memory_policies": [b["policy"] for b in with_memory["breaches"]],
                    "control_policies": [b["policy"] for b in control["breaches"]],
                    "difference": control["breach_count"] - with_memory["breach_count"],
                    "compliance_difference": round(stats.differences[-1], 2),
                })

                report(display.row_for(stats, index + 1, len(selection), task.id))
                # Retries and errors keep to their own lines rather than
                # lengthening the pair line and breaking the columns.
                # An unreadable review no longer reaches here: the pair was
                # excluded above, before it could be scored.
                problems = sorted({*with_memory["errors"], *control["errors"]})
                for problem in problems:
                    report(display.note(problem))
                failing_streak = failing_streak + 1 if problems else 0
                _write_live_html(html_path, run_id, config, providers, scenario,
                                 records, stats, live=True)

                if config.error_streak_limit and failing_streak >= config.error_streak_limit:
                    abandoned = {
                        "after_pair": index + 1,
                        "consecutive_failures": failing_streak,
                        "reason": problems[0] if problems else "repeated errors",
                    }
                    report(display.abandoned(failing_streak, abandoned["reason"]))
                    break

                if index + 1 < len(selection):
                    _pace_floor(config.pace_seconds, monotonic() - since_memory, sleep)
        except KeyboardInterrupt:
            # Stopping early is a legitimate way to use this: every pair already
            # written is complete and the report reads whatever is there.
            interrupted = True
            report(f"\ninterrupted after {stats.pairs} pair(s); results are on disk")

    # However the loop ended, the page left behind is finished: no refresh tag,
    # so it is a single self-contained file somebody can send on.
    _write_live_html(html_path, run_id, config, providers, scenario,
                     records, stats, live=False, abandoned=abandoned)
    return records, stats, interrupted, abandoned, excluded_review


def _complete_pairs(jsonl_path: Path) -> list[dict[str, Any]]:
    """The episode records of every pair that actually measured something.

    Three ways a pair can be on disk without being a measurement, and all are
    dropped here so a resumed run does the task again rather than inheriting
    it. A crash can leave one arm with nothing to compare it to. An episode
    that failed outright was recorded as failed with a breach count of zero,
    which is not a good episode with nothing wrong. And a review that could not
    be read leaves its episode with no verdict at all, recorded the same way.
    Carrying any of them forward would both skip a task that was never scored
    and feed a fabricated zero into the comparison, which is precisely what the
    live loop refuses to do.
    """
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    arms: dict[int, set[str]] = {}
    failed: set[int] = set()
    for record in records:
        index = record.get("pair_index", -1)
        arms.setdefault(index, set()).add(record.get("arm", ""))
        if record.get("episode_failed") or record.get("review_failed"):
            failed.add(index)
    measured = {
        index
        for index, seen in arms.items()
        if {"memory", "control"} <= seen and index not in failed
    }
    return [r for r in records if r.get("pair_index") in measured]


def _write_live_html(
    path: Path,
    run_id: str,
    config: RunConfig,
    providers: _Providers,
    scenario: Scenario,
    records: list[dict[str, Any]],
    stats: PairStats,
    live: bool,
    abandoned: dict[str, Any] | None = None,
) -> None:
    """Re-render the page after every pair.

    Rendering from scratch each time rather than patching means the page a
    reader opens mid-run is the same page they would get at the end, and an
    interrupted run leaves a correct file rather than a half-updated one. It
    costs a few milliseconds against episodes that take seconds.

    Imported here rather than at the top because the report module reads the
    run's own output, and the runner should not depend on the reporter to run.
    """
    from .report import Run, write_html

    meta = {
        "run_id": run_id,
        "seed": None if config.all_variants else config.seed,
        "all_variants": config.all_variants,
        "paired": True,
        "memory_enabled": True,
        "pairing": stats.as_record(),
        "abandoned": abandoned,
        "grader_version": GRADER_VERSION,
        "reference_date": scenario.reference_date,
        **providers.as_record(),
    }
    whole = Run(run_id=run_id, meta=meta, records=tuple(records), path=path)
    try:
        write_html(whole.as_arm("memory"), whole.as_arm("control"), path, live=live)
    except (OSError, ValueError) as error:  # noqa: BLE001 - a page is not the run
        # The run is the point; the page is a view of it. A rendering fault
        # must never take down an episode loop that is spending real money.
        print(f"live report not written: {error}")


def _report_pairs(
    stats: PairStats,
    interrupted: bool,
    report: Callable[[str], None],
    abandoned: dict[str, Any] | None = None,
    excluded_review: int = 0,
) -> None:
    """The closing summary, in the same plain words as the table.

    No statistics vocabulary here either. A reader who wants the endpoints has
    them in the JSONL and in the HTML report; a reader watching a run wants to
    know what happened, and "20 points ahead, give or take 4" is that, where
    "mean d = 20.4, 95% CI [16.2, 24.6]" is the same fact addressed to somebody
    who already knew.
    """
    if not stats.pairs:
        report("no tasks completed")
        return
    interval = stats.interval()
    report("")
    report(
        f"{stats.pairs} task(s), each answered twice"
        + (", stopped early" if interrupted else "")
    )
    report(f"  without memory  {stats.control_compliance:.0f}% of applicable "
           "policies met")
    report(f"  with memory     {stats.memory_compliance:.0f}% of applicable "
           "policies met")
    if interval is not None:
        half = (interval.high - interval.low) / 2
        report(f"  memory advantage {interval.mean:.0f} ±{half:.0f} "
               "percentage points")
    report(f"  underneath: {sum(stats.control_breaches) / stats.pairs:.2f} rule "
           f"breaches per task without memory, "
           f"{sum(stats.memory_breaches) / stats.pairs:.2f} with")
    if excluded_review:
        report(f"  {excluded_review} pair(s) excluded: review failed")
    drift = stats.control_drift()
    if drift is not None and abs(drift) >= 2:
        report(f"  {display.drift_note(drift)}")
    if abandoned:
        report(
            f"  this run was abandoned after {abandoned['after_pair']} task(s): "
            f"{abandoned['consecutive_failures']} in a row hit errors "
            f"({abandoned['reason']}). These figures are diagnostic only."
        )
    report(
        "  a run can be stopped at any point and this summary still reads; the "
        "figures quoted in this repository's documentation come from complete "
        "runs of a fixed length."
    )


def _pace_floor(
    seconds: float, already_spent: float, sleeper: Callable[[float], None]
) -> None:
    """Hold the gap as a floor rather than an addition.

    The single-arm run sleeps the whole gap because nothing happens in it. A
    paired run has already spent the control episode inside that window, so only
    the remainder is waited for, and a slow control episode means no wait at all.
    """
    remaining = seconds - already_spent
    if remaining > 0:
        sleeper(remaining)


def _run_task(
    index: int,
    run_id: str,
    task: Task,
    variant_index: int,
    scenario: Scenario,
    providers: _Providers,
    reviewer: Reviewer,
    reflector: Reflector | None,
    memory: MemcoClient | None,
    arm: str = "memory",
    pair_index: int | None = None,
) -> dict[str, Any]:
    variant = task.variants[variant_index]
    account = scenario.account_for(task)
    order = scenario.order_for(task)

    started_at = stamp()
    # One session for the whole episode. The searches the agent makes while
    # drafting, the ratings those searches earn, and the lessons written from
    # the correction afterwards are all one task's work, and naming the session
    # on each of them is what records them as one task's work rather than as
    # unrelated calls that happen to be adjacent. The blind arm has no memory,
    # and so no session.
    session = memory.open_session() if memory is not None else None
    draft = draft_reply(task, variant, scenario, providers.agent, session)
    review = reviewer.review(task, variant, draft.text, account, order)
    reflection = (
        reflector.reflect(task, variant, draft, review, scenario.policies, session)
        if reflector and session is not None and not review.failed
        else Reflection()
    )

    errors: list[str] = []
    if session is not None and session.error:
        errors.append(session.error)
    errors += [*draft.memory_errors, *reflection.errors]
    if review.error:
        errors.append(f"reviewer: {review.error}")
    if draft.truncated:
        errors.append("agent: draft hit the output token limit")

    record: dict[str, Any] = {
        "run_id": run_id,
        "task_index": index,
        "task_id": task.id,
        "variant_index": variant_index,
        "memory_enabled": memory is not None,
        "arm": arm,
        "pair_index": pair_index,
        "applicable_policies": list(task.applicable_policies),
        "breaches": [asdict(breach) for breach in review.breaches],
        "breach_count": review.breach_count,
        "review_failed": review.failed,
        "dropped_policies": list(review.dropped),
        # The lesson text is kept here, not just its title: these lines are the
        # draft policy book, and it has to be readable without a second call to
        # the server.
        "lessons_written": [
            {
                "title": lesson.title,
                "op_id": lesson.op_id,
                "query": lesson.query,
                "content": lesson.content,
                "policy": lesson.policy,
                "at": lesson.at,
            }
            for lesson in reflection.lessons_written
            if lesson.error is None
        ],
        "lessons_retrieved": [
            {"title": insight.title, "memory_idx": insight.memory_idx, "idx": insight.idx}
            for insight in draft.retrieved
        ],
        # Every search, hit or miss, with the query and the moment it went out.
        # A miss is the interesting half: it dates how long a lesson written
        # earlier had still not become searchable.
        "searches": [
            {
                "query": search.query,
                "at": search.at,
                "hits": len(search.insights),
                "session_id": search.session_id,
                "error": search.error,
            }
            for search in draft.searches
        ],
        # `policy` is the attribution: which policy the retrieved lesson was
        # judged to bear on. It is what lets the report separate a breach the
        # agent had been told about from one it had not.
        "feedback_sent": [
            {
                "idx": entry.idx,
                "relevant": entry.relevant,
                "correct": entry.correct,
                "comment": entry.comment,
                "policy": entry.policy,
            }
            for entry in reflection.feedback_sent
        ],
        # What the server said to each share_feedback call. A write proves it
        # landed with an operation id; feedback answers in prose, and without it
        # "no error" is the only evidence the call did anything.
        "feedback_calls": [
            {
                "session_id": call.session_id,
                "entries": call.entries,
                "detail": call.detail,
                "at": call.at,
            }
            for call in reflection.feedback_calls
        ],
        "draft": draft.text,
        "corrected_draft": review.corrected_draft,
        **providers.as_record(),
        "grader_version": GRADER_VERSION,
        "started_at": started_at,
        "finished_at": stamp(),
        "errors": errors,
    }
    if review.failed and review.raw:
        record["review_raw"] = review.raw
    return record


def _pace(seconds: float, memory_on: bool, sleeper: Callable[[float], None]) -> None:
    """Hold a gap between episodes. The reason is twofold.

    It is an indexing allowance. The server accepts a write and processes it
    asynchronously, so a lesson written on one task needs a moment before the
    next task can retrieve it; measured write-to-searchable time is under five
    seconds. Human-paced work gets that gap for free from the space between
    tasks, and a harness running tasks back to back has to put it back.

    It also keeps the run inside the server's rate limit. An unpaced run trips
    the limit partway through and starts drafting episodes with no memory at
    all, which reads as an agent that cannot learn rather than a client that
    was asked to slow down.

    A run with memory off makes no memory calls, so it has nothing to pace and
    waits for nothing. Set the gap to zero to turn it off everywhere.
    """
    if seconds > 0 and memory_on:
        sleeper(seconds)


def _select(scenario: Scenario, config: RunConfig) -> list[tuple[Task, int]]:
    """The episodes to run, in order."""
    if config.all_variants:
        # Every task, every variant, in file order. No sampling, no seed: this
        # mode measures the scenario, so every email has to appear exactly once.
        return [
            (task, index)
            for task in scenario.tasks
            for index in range(len(task.variants))
        ]
    return _sample(scenario, config)


def _sample(scenario: Scenario, config: RunConfig) -> list[tuple[Task, int]]:
    """Pick tasks without replacement, then a variant for each, from one seed."""
    rng = random.Random(config.seed)
    count = min(config.task_count, len(scenario.tasks))
    tasks = rng.sample(list(scenario.tasks), k=count)
    return [(task, rng.randrange(len(task.variants))) for task in tasks]


def _run_id(started: datetime, config: RunConfig) -> str:
    stamp = started.strftime("%Y-%m-%dT%H%M%S")
    selection = "allvariants" if config.all_variants else f"seed{config.seed}"
    label = f"-{_slug(config.label)}" if config.label.strip() else ""
    suffix = "" if config.memory_enabled else "-nomemory"
    return f"{stamp}-{selection}{label}{suffix}"


def _slug(label: str) -> str:
    """Keep a caller's label to characters that are safe in a filename."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return cleaned or "run"


def _unique_run_id(run_id: str, results_dir: Path) -> str:
    """Never overwrite an earlier run.

    The id is stamped to the second, so two runs started in the same second
    would otherwise write to the same file and the first would be lost.
    """
    candidate, attempt = run_id, 1
    while (results_dir / f"{candidate}.jsonl").exists():
        attempt += 1
        candidate = f"{run_id}-{attempt}"
    return candidate


def _write_meta(
    path: Path,
    run_id: str,
    config: RunConfig,
    providers: _Providers,
    scenario: Scenario,
    started: datetime,
    episode_count: int,
    finished: datetime | None = None,
    total_breaches: int | None = None,
    pairing: dict[str, Any] | None = None,
    interrupted: bool = False,
    abandoned: dict[str, Any] | None = None,
) -> None:
    meta: dict[str, Any] = {
        "run_id": run_id,
        "seed": None if config.all_variants else config.seed,
        "task_count": episode_count,
        "requested_task_count": config.task_count,
        "all_variants": config.all_variants,
        "label": config.label,
        "pace_seconds": config.pace_seconds,
        "memory_enabled": config.memory_enabled,
        "paired": config.paired,
        **providers.as_record(),
        "reference_date": scenario.reference_date,
        "policy_count": len(scenario.policies),
        "library_size": len(scenario.tasks),
        "grader_version": GRADER_VERSION,
        "started_at": started.isoformat(),
    }
    if finished is not None:
        meta["finished_at"] = finished.isoformat()
        meta["usage"] = providers.usage().as_record()
        by_role = providers.usage_by_role()
        if by_role:
            meta["usage_by_role"] = by_role
        adaptations = providers.adaptations()
        if adaptations:
            meta["provider_adaptations"] = adaptations
    if pairing is not None:
        meta["pairing"] = pairing
        meta["episodes"] = episode_count * 2
    if interrupted:
        meta["interrupted"] = True
    if abandoned:
        # The one flag a reader must not miss: the run stopped because it had
        # stopped measuring anything.
        meta["abandoned"] = abandoned
    if total_breaches is not None:
        meta["total_breaches"] = total_breaches
        meta["mean_breaches"] = round(total_breaches / episode_count, 4) if episode_count else 0.0
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
