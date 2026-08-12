"""The running statistics of a paired run.

Each task is answered twice, once by an agent with memory and once by an agent
without, from the same email. The two answers differ only in whether memory was
available, so their difference is a measurement of memory rather than of the
task, and the tasks themselves cancel out. That is the whole reason for pairing:
tasks vary enormously in how many policies they can breach, and comparing two
separate runs means comparing two different draws from that variation.

What is scored is **compliance**: the share of the policies that applied to a
task that the reply actually got right. Higher is better. A bare breach count
says how many things went wrong but not out of how many, and tasks here carry
anywhere from two to six applicable policies, so the same count of two means a
good reply on one task and a poor one on another. The counts are still recorded
per task, and still printed; they are just not the quantity the trend is read
from.

Trends are **pooled, never averaged**. Compliance over a window of tasks is the
obligations met across the whole window divided by the obligations that applied
across it. Averaging per-task percentages instead would give a two-policy task
the same vote as a six-policy one, and on two policies the only values that
exist are 0, 50 and 100, so the line would jump between them and every jump
would look like news. Pooling ten tasks puts around forty obligations in the
denominator, and the line moves in steps of roughly two and a half points.

What is tracked per pair is `d`, the memory arm's compliance minus the control
arm's, in percentage points. Positive means memory did better. The confidence
interval is the ordinary t interval on those differences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["Interval", "PairStats", "compliance", "pooled", "running_intervals"]

# Two-sided 95% critical values of Student's t. A table rather than a formula
# because it can be checked against any statistics text, and the whole of what
# is needed is one column of it.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}
_T_INFINITY = 1.960

# Before this many pairs the interval is too wide to mean anything, and putting
# a figure on three tasks would be a claim the data cannot carry. Until then the
# advantage column prints an em-dash and the chart draws the line unbanded.
ADVANTAGE_FLOOR = 10

# The window for the trailing compliance. Ten is the same window the learning
# curve is smoothed over, so the live line and the finished chart agree.
TRAILING = 10


def t_critical(df: int) -> float:
    """The two-sided 95% critical value, interpolated between tabulated rows."""
    if df < 1:
        return _T_INFINITY
    if df in _T_95:
        return _T_95[df]
    keys = sorted(_T_95)
    if df > keys[-1]:
        return _T_INFINITY
    above = min(key for key in keys if key > df)
    below = max(key for key in keys if key < df)
    span = above - below
    return _T_95[below] + (_T_95[above] - _T_95[below]) * (df - below) / span


def compliance(breaches: int, applicable: int) -> float:
    """One task's score: the share of the policies that applied that it met."""
    if applicable <= 0:
        # Nothing applied, so nothing could be got wrong. Tasks like this do not
        # occur in the shipped scenario; the guard is here so a scenario that
        # grows one cannot divide by zero halfway through somebody's run.
        return 100.0
    return 100.0 * (applicable - breaches) / applicable


def pooled(breaches: Sequence[int], applicable: Sequence[int]) -> float:
    """Compliance over several tasks at once, on one shared denominator.

    Every obligation across the window counts once, whichever task it came from.
    This is what makes a task with six policies at stake weigh more than a task
    with two, which is the honest weighting: it had more to get wrong.
    """
    total = sum(applicable)
    if total <= 0:
        return 100.0
    return 100.0 * (total - sum(breaches)) / total


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float


def running_intervals(differences: list[float]) -> list[tuple[float, float | None]]:
    """The running mean of the differences, and its half-width, after each pair.

    This is the memory advantage: the same quantity the terminal prints in its
    own column, drawn on the chart as a line with a band. The half-width is
    None while there is too little to say, and the band simply does not start
    until there is.
    """
    out: list[tuple[float, float | None]] = []
    for count in range(1, len(differences) + 1):
        window = differences[:count]
        mean = sum(window) / count
        if count < ADVANTAGE_FLOOR:
            out.append((mean, None))
            continue
        variance = sum((d - mean) ** 2 for d in window) / (count - 1)
        half = t_critical(count - 1) * math.sqrt(variance / count) if variance > 0 else 0.0
        out.append((mean, half))
    return out


@dataclass
class PairStats:
    """Everything the live line and the final report need, updated per pair."""

    differences: list[float] = field(default_factory=list)  # percentage points
    memory_breaches: list[int] = field(default_factory=list)
    control_breaches: list[int] = field(default_factory=list)
    # Both arms answered the same task, so one applicable count serves the pair.
    applicable: list[int] = field(default_factory=list)

    def add(self, memory: int, control: int, applicable: int) -> None:
        self.memory_breaches.append(memory)
        self.control_breaches.append(control)
        self.applicable.append(applicable)
        self.differences.append(
            compliance(memory, applicable) - compliance(control, applicable)
        )

    @property
    def pairs(self) -> int:
        return len(self.differences)

    @property
    def memory_compliance(self) -> float:
        """The memory arm's compliance over the whole run so far."""
        return pooled(self.memory_breaches, self.applicable)

    @property
    def control_compliance(self) -> float:
        return pooled(self.control_breaches, self.applicable)

    def interval(self) -> Interval | None:
        """The t interval on the paired differences, or None while it is undefined."""
        n = len(self.differences)
        if n < 2:
            return None
        mean = sum(self.differences) / n
        variance = sum((d - mean) ** 2 for d in self.differences) / (n - 1)
        if variance <= 0:
            # Every pair differed by the same amount. The interval is a point,
            # which is honest: there is no spread to widen it with.
            return Interval(mean=mean, low=mean, high=mean)
        margin = t_critical(n - 1) * math.sqrt(variance / n)
        return Interval(mean=mean, low=mean - margin, high=mean + margin)

    def trailing(self, breaches: list[int], window: int = TRAILING) -> float:
        """Pooled compliance for one arm over the last `window` pairs."""
        return pooled(breaches[-window:], self.applicable[-window:])

    def control_drift(self) -> float | None:
        """How far the control arm's compliance moved, first half to last.

        The control agent cannot learn, so its compliance is expected to sit
        flat. If it does not, the tasks drawn later were easier or harder than
        the ones drawn early, and the memory arm's climb is partly that. It is
        reported rather than corrected for: correcting would mean modelling the
        drift, and the honest thing is to say the sample moved.
        """
        if len(self.control_breaches) < 2 * TRAILING:
            return None
        half = len(self.control_breaches) // 2
        first = pooled(self.control_breaches[:half], self.applicable[:half])
        last = pooled(self.control_breaches[half:], self.applicable[half:])
        return last - first

    def as_record(self) -> dict[str, object]:
        interval = self.interval()
        record: dict[str, object] = {
            "pairs": self.pairs,
            # Percentages and percentage points: the headline quantities.
            "memory_compliance": round(self.memory_compliance, 2),
            "control_compliance": round(self.control_compliance, 2),
            "mean_difference": round(sum(self.differences) / self.pairs, 2)
            if self.pairs
            else 0.0,
            # And the raw counts underneath them, so nothing is lost for
            # analysis by the report choosing to lead with a percentage.
            "memory_mean": round(sum(self.memory_breaches) / self.pairs, 4)
            if self.pairs
            else 0.0,
            "control_mean": round(sum(self.control_breaches) / self.pairs, 4)
            if self.pairs
            else 0.0,
            "applicable_total": sum(self.applicable),
        }
        if interval is not None:
            record["confidence_interval"] = [round(interval.low, 2), round(interval.high, 2)]
        drift = self.control_drift()
        if drift is not None:
            record["control_drift"] = round(drift, 2)
        return record
