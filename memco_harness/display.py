"""What a paired run shows while it runs.

Formatting only: nothing here talks to a model, a file, or the clock. The run
loop hands it numbers and it returns strings, which is what makes the columns
checkable in a test rather than by squinting at a terminal.

The rules it follows are worth stating, because they are the difference between
a table somebody can read at a glance and a wall of digits:

Labels live in the two header rows and never in the lines, so identical
quantities line up into columns the eye can run down. No-mem comes before mem
everywhere, in the header, in the rows, and in the chart, so the reader never
has to check which way round a pair is. And no statistics vocabulary appears at
all: "memory advantage 20 ±9" says what happened, where "mean d = 20.4, 95% CI
[11.6, 29.2]" says the same thing to a reader who already knows what it means.
The endpoints are in the JSONL and the HTML for anyone who wants the working.

`memory advantage` is the name of one quantity — the running mean of the paired
compliance difference — and it is the name everywhere: this column, the closing
summary, and the line and band on the report's chart. One quantity, one name, so
nobody has to work out whether two figures are the same thing.

There is no event line for it. A milestone printed the moment an interval
cleared zero made one arbitrary pair look like the finding, when the finding is
the column itself: the advantage and its ± are on every line from pair ten, and
a reader can see for themselves when the interval sits clear of zero.
"""

from __future__ import annotations

from .pairing import ADVANTAGE_FLOOR, PairStats

__all__ = ["abandoned", "configuration", "drift_note", "header", "note",
           "pair_row", "preamble", "row_for"]

PREAMBLE_TEMPLATE = """\
Learning on the job — paired run over {tasks} tasks
Each task is answered twice: with shared memory (mem) and without
(no-mem). Each reply is scored on the share of applicable policies it
gets right (compliance); higher is better. The memory arm learns from
the reviewer's corrections as it goes, so watch its compliance climb
while no-mem stays put."""

# Column layout, in two rows: the top names the measures as groups, the second
# splits each group into its two arms. The header is the specification: every
# row is built to sit under it, so changing one means changing the other.
HEADER_GROUPS = (
    "                    violations (#)   compliance 10-avg (%)   "
    "memory advantage (%)"
)
HEADER_COLUMNS = " pair    task       no-mem    mem      no-mem     mem"

# Fixed: the header is a constant, so the row has to be one too.
_PAIR_WIDTH = 8  # "   4/100", " 200/200"
_TASK_WIDTH = 9  # task-0042
_VIOLATIONS_NO_MEM_WIDTH = 6
_VIOLATIONS_MEM_WIDTH = 8
_COMPLIANCE_NO_MEM_WIDTH = 11
_COMPLIANCE_MEM_WIDTH = 10
# The advantage is right-aligned in this field and the em-dash sits in the same
# one, so the digits and the "nothing yet" marker share a column and the eye can
# run straight down it. The ± trails after, outside the aligned part.
_ADVANTAGE_WIDTH = 16

# A hyphen in this column would read as a minus sign against a number that can
# genuinely go negative, so the "not yet" marker is an em-dash.
NOT_YET = "—"


def preamble(tasks: int) -> str:
    """The standing explanation, printed at the start and atop the final report."""
    return PREAMBLE_TEMPLATE.format(tasks=tasks)


def configuration(seed: int | None, agent: str, reviewer: str) -> str:
    """One line of provenance, the only other prose before the table."""
    where = f"seed {seed}" if seed is not None else "all variants"
    return f"{where} · drafting {agent} · reviewing {reviewer}"


def header() -> str:
    return f"{HEADER_GROUPS}\n{HEADER_COLUMNS}"


def pair_row(
    index: int,
    total: int,
    task_id: str,
    no_mem: int,
    mem: int,
    compliance_no_mem: float,
    compliance_mem: float,
    advantage: float | None,
    half_width: float | None,
) -> str:
    """One completed pair.

    The two compliance columns are pooled over the last ten pairs, or over the
    pairs that exist while there are fewer than ten: a share of obligations met
    reads correctly from the first pair, however few are behind it.

    The advantage is the one column that has to wait. It is an interval, and an
    interval over a handful of pairs is noise in the costume of precision, so it
    prints an em-dash until there are ten.
    """
    counter = f"{index}/{total}".rjust(_PAIR_WIDTH)
    if advantage is None or half_width is None:
        ahead = f"{NOT_YET:>{_ADVANTAGE_WIDTH}}"
    else:
        ahead = f"{advantage:>{_ADVANTAGE_WIDTH}.0f} ±{half_width:.0f}"
    return (
        f"{counter}"
        f" {task_id:<{_TASK_WIDTH}}"
        f"{no_mem:>{_VIOLATIONS_NO_MEM_WIDTH}}"
        f"{mem:>{_VIOLATIONS_MEM_WIDTH}}"
        f"{compliance_no_mem:>{_COMPLIANCE_NO_MEM_WIDTH}.0f}"
        f"{compliance_mem:>{_COMPLIANCE_MEM_WIDTH}.0f}"
        f"{ahead}"
    )


def drift_note(drift: float) -> str:
    """What a moving no-memory line means, in the direction it actually moved.

    The control agent cannot learn, so its compliance should sit flat. When it
    does not, the sample moved under the run, and which way matters: compliance
    that rises means the later tasks were easier, and some of the memory arm's
    climb is the sample rather than the memory; compliance that falls means they
    were harder, which understates whatever memory did. Saying it the wrong way
    round would tell a reader the result is flattering when it is conservative.
    """
    # The direction is in the verb, so the figure is a magnitude: "fell -3" reads
    # as a double negative and makes a reader stop to work out which way it went.
    points = abs(drift)
    if drift > 0:
        return (
            f"the no-memory line rose {points:.0f} points between the halves of this "
            "run, so the tasks drawn later were easier than those drawn early and "
            "some of the improvement belongs to the sample rather than to memory. "
            "Noted, not corrected for."
        )
    return (
        f"the no-memory line fell {points:.0f} points between the halves of this run, "
        "so the tasks drawn later were harder than those drawn early and the gap "
        "memory opened is if anything understated. Noted, not corrected for."
    )


def abandoned(streak: int, reason: str) -> str:
    """Said loudly, because the run has stopped being a measurement.

    A memory call that fails is recorded and the task carries on, which is right
    for a hiccup. A run of them is a different thing: the memory arm is drafting
    with nothing to draw on, so it has quietly become a second control arm while
    the table keeps printing numbers that look like a comparison. Better to stop
    and say so than to spend an hour producing a figure nobody should quote.
    """
    # Which failure it was decides what to say, and the two are not the same
    # problem. `memco_client` prefixes everything it could not do with "memco",
    # so a reason that starts there is the memory server; anything else is the
    # episode itself falling over, and blaming memory for it would send the next
    # person to the wrong place.
    diagnosis = (
        "The memory arm cannot reach its memory, so it is no longer being\n"
        "     compared with anything."
        if reason.lower().startswith("memco")
        else "The episodes are failing outright, so there is nothing being\n"
        "     compared either way."
    )
    return (
        f"\n  ── stopped: {streak} tasks in a row hit errors ──\n"
        f"     {reason}\n"
        f"     {diagnosis} What is on disk up to this point is sound;\n"
        "     the run as a whole is not a measurement and should not be quoted."
    )


def note(text: str) -> str:
    """A retry, a rate limit, an error. Never lengthens a pair line."""
    return f"    {text}"


def row_for(stats: PairStats, index: int, total: int, task_id: str) -> str:
    """The row for the pair just added to `stats`."""
    interval = stats.interval() if stats.pairs >= ADVANTAGE_FLOOR else None
    return pair_row(
        index=index,
        total=total,
        task_id=task_id,
        no_mem=stats.control_breaches[-1],
        mem=stats.memory_breaches[-1],
        compliance_no_mem=stats.trailing(stats.control_breaches),
        compliance_mem=stats.trailing(stats.memory_breaches),
        advantage=interval.mean if interval else None,
        half_width=(interval.high - interval.low) / 2 if interval else None,
    )
