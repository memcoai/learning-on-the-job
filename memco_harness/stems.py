"""The two questions memory is written under and read back with.

A lesson is only worth writing if a search will later find it, and whether it
does is decided by one thing: whether the phrasing it was filed under resembles
the phrasing somebody looks for. Left to itself, each side words the question
its own way, and the two drift apart — a run of this harness wrote fifty-six
lessons about how a reply opens, under queries no drafting agent ever asked, and
retrieved none of them across ninety well-aimed searches for exactly that.

So the two sides are given the same literal text. The write side fills a stem to
file a lesson; the read side fills the same stem to look one up. The stems carry
no scenario or policy vocabulary of their own: everything specific arrives in
the angle-bracket tail, written at the time by whoever is filling it.

Two stems, because knowledge about work divides in two. What to decide is one
kind, and a question about a situation finds it. How the work is done is the
other, and no situational question ever surfaces it however well it is stored.
"""

from __future__ import annotations

__all__ = ["CRAFT_STEM", "SITUATION_STEM"]

# The shape and conventions of the thing being produced.
CRAFT_STEM = "How are replies from this desk written and structured when <kind of reply>?"

# The decision the situation calls for.
SITUATION_STEM = "What applies when a customer <request>, and <relevant circumstances>?"
