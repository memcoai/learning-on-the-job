"""What a paired run shows while it runs.

The display is a specification, not a preference: the columns are what make a
run readable at a glance, and a change that quietly widens one is a regression
nobody would notice until they were watching a real run. So the layout is
asserted against the sample it was designed from.
"""

from __future__ import annotations

import re

from memco_harness import display
from memco_harness.pairing import ADVANTAGE_FLOOR, PairStats

# Terms that say the same thing to a reader who already knows what they mean,
# and nothing at all to the reader this display is for.
BANNED = ("CI", "confidence", "interval", "mean d", "p-value", "significant",
          "std", "variance", "t-test")


def columns() -> str:
    """The second header row: the one the numbers sit under."""
    return display.header().splitlines()[1]


def groups() -> str:
    """The first header row: the measures, as merged column groups."""
    return display.header().splitlines()[0]


def test_a_row_matches_the_agreed_layout_exactly():
    """The layout the display was designed against, once every column has a
    number in it. The run-up is covered separately: the advantage holds an
    em-dash until ten pairs exist, which the original sample predated."""
    assert display.pair_row(
        index=15, total=100, task_id="task-0060", no_mem=1, mem=1,
        compliance_no_mem=60, compliance_mem=80, advantage=21, half_width=8,
    ) == "  15/100 task-0060     1       1         60        80              21 ±8"


def test_the_compliance_columns_never_wait():
    """A share of obligations met reads correctly from the first pair, however
    few are behind it, so unlike an interval it has nothing to wait for."""
    early = display.pair_row(4, 100, "task-0025", 2, 1, 50, 75, None, None)
    assert early == "   4/100 task-0025     2       1         50        75               —"
    assert "50" in early and "75" in early


def test_every_column_sits_under_its_heading():
    header = columns()
    row = display.pair_row(4, 100, "task-0025", 2, 1, 50, 75, None, None)
    violations_no_mem = header.index("no-mem")
    violations_mem = header.index("mem", violations_no_mem + len("no-mem"))
    compliance_no_mem = header.index("no-mem", violations_mem)
    compliance_mem = header.index("mem", compliance_no_mem + len("no-mem"))
    # Every value is right-aligned in its field, so it is the last character
    # that has to land under the heading. Two-digit values start one earlier.
    for end, label_at, label in (
        (23, violations_no_mem, "no-mem"),
        (31, violations_mem, "mem"),
        (42, compliance_no_mem, "no-mem"),
        (52, compliance_mem, "mem"),
    ):
        assert label_at <= end <= label_at + len(label), f"{label} at {label_at}"
    assert row[18:24].strip() == "2" and row[24:32].strip() == "1"
    assert row[32:43].strip() == "50" and row[43:53].strip() == "75"
    # no-mem is read before mem, in both header rows and in the row.
    assert violations_no_mem < violations_mem < compliance_no_mem < compliance_mem
    assert groups().index("violations") < groups().index("compliance")


def test_the_two_header_rows_name_the_measures_and_then_split_them():
    top, bottom = groups(), columns()
    assert "violations (#)" in top and "compliance 10-avg (%)" in top
    # The advantage is a paired quantity, so it has no per-arm split beneath it.
    assert "memory advantage (%)" in top
    assert bottom.count("no-mem") == 2 and bottom.rstrip().endswith("mem")
    assert "advantage" not in bottom


def test_the_marker_before_ten_pairs_is_an_em_dash_not_a_hyphen():
    """A hyphen would read as a minus sign in a column that goes negative."""
    row = display.pair_row(4, 100, "task-0025", 2, 1, 50, 75, None, None)
    assert row.rstrip().endswith("—")
    assert not row.rstrip().endswith("-")


def test_a_negative_advantage_needs_no_special_casing():
    """Memory being behind is a real outcome and prints like any other."""
    row = display.pair_row(96, 100, "task-0007", 0, 2, 70, 65, -5, 4)
    assert "-5 ±4" in row


def test_the_advantage_and_the_em_dash_share_a_column():
    """So the eye can run down one column rather than hunting for the number."""
    waiting = display.pair_row(4, 100, "task-0025", 2, 1, 50, 75, None, None)
    ready = display.pair_row(15, 100, "task-0060", 1, 1, 60, 80, 21, 8)
    assert waiting.index("—") == ready.index("21") + len("21") - 1


def test_labels_appear_only_in_the_header():
    row = display.pair_row(10, 100, "task-0141", 3, 0, 58, 78, 20, 9)
    for label in ("no-mem", "mem", "10-avg", "advantage", "violations", "pair", "task "):
        assert label not in row, f"{label!r} repeated in the row breaks the columns"


def test_the_variant_index_is_provenance_and_stays_out_of_the_terminal():
    row = display.pair_row(10, 100, "task-0141", 3, 0, 58, 78, 20, 9)
    assert not re.search(r"/v\d", row)


def test_no_statistics_vocabulary_reaches_the_terminal():
    surfaces = [
        display.preamble(100),
        display.header(),
        display.pair_row(10, 100, "task-0141", 3, 0, 58, 78, 20, 9),
        display.note("memory search retried after a rate limit"),
        display.configuration(42, "anthropic:x", "anthropic:y"),
    ]
    # Whole words: "CI" the abbreviation is banned, the "ci" inside "policies"
    # and "compliance" is not, and a substring match cannot tell them apart.
    for text in surfaces:
        for term in BANNED:
            found = re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE)
            assert not found, f"{term!r} in {text!r}"


def test_there_is_no_milestone_event_line():
    """One arbitrary pair printed as an event made the crossing look like the
    finding, when the finding is the advantage column and its ± on every line."""
    assert not hasattr(display, "milestone")


def test_a_note_is_indented_and_never_lengthens_a_pair_line():
    assert display.note("rate limited, waiting").startswith("    ")


def test_the_row_reads_its_numbers_off_the_running_statistics():
    stats = PairStats()
    for _ in range(ADVANTAGE_FLOOR - 1):
        stats.add(memory=1, control=3, applicable=4)
    early = display.row_for(stats, stats.pairs, 100, "task-0011")
    assert early.rstrip().endswith("—"), "too few pairs to put an interval on it"
    assert early.count("—") == 1, "only the advantage waits; compliance is there already"
    assert " 75" in early and " 25" in early, "both arms' compliance, no-mem first"

    stats.add(memory=1, control=3, applicable=4)
    ready = display.row_for(stats, stats.pairs, 100, "task-0012")
    assert "—" not in ready, "at ten pairs every column has something to say"
    assert "50 ±0" in ready, "ten identical pairs: fifty points ahead, with no spread"
    assert " 3 " in ready and " 1 " in ready, "this pair's raw counts, no-mem first"


def test_the_preamble_says_what_is_being_watched_without_jargon():
    text = display.preamble(100)
    assert "paired run over 100 tasks" in text
    assert "higher is better" in text
    assert "compliance" in text
    assert "no-mem" in text and "mem" in text


def test_the_columns_hold_still_whatever_the_task_count():
    """The header is a constant, so the rows have to be. Sizing the pair counter
    to the digits in the total made a 30-task run two columns narrower than a
    100-task one, and every column slid out from under its heading."""
    header = columns()
    for total in (9, 30, 100, 200):
        row = display.pair_row(1, total, "task-0164", 2, 1, 50, 75, None, None)
        assert row.index("task-0164") == header.index("task"), f"total={total}"
        assert row.rstrip().endswith("—"), f"total={total}"
    # And the counter itself stays inside its column at both ends of a run.
    assert display.pair_row(200, 200, "task-0083", 2, 0, 60, 80, 21, 8).index(
        "task-0083"
    ) == header.index("task")


def test_the_drift_note_says_which_way_the_sample_moved():
    """The no-memory arm cannot learn, so moving compliance means the sample
    moved. Which way decides whether the result is flattered or understated, and
    saying it the wrong way round would tell a reader the opposite of the truth.
    Compliance rising means easier tasks, which is the reverse of what a rising
    breach count meant, so this is the assertion that catches a stale flip."""
    rising = display.drift_note(6.7)
    assert "easier" in rising and "belongs to the sample" in rising
    assert "harder" not in rising

    falling = display.drift_note(-6.7)
    assert "harder" in falling and "understated" in falling
    assert "easier" not in falling

    for text in (rising, falling):
        assert "not corrected for" in text, "reported, never adjusted away"


def test_the_stop_message_names_the_failure_it_actually_saw():
    """A memory arm that cannot reach memory and a model that cannot be reached
    at all are both reasons to stop, and they are not the same reason. Saying
    the wrong one sends the next person to the wrong place."""
    lost_memory = display.abandoned(3, "memco search failed: daily search limit")
    assert "cannot reach its memory" in lost_memory

    lost_model = display.abandoned(3, "memory episode failed: APIConnectionError")
    assert "cannot reach its memory" not in lost_model
    assert "failing outright" in lost_model

    for text in (lost_memory, lost_model):
        assert "should not be quoted" in text
