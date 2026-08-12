"""The contract between what memory is written under and what it is read with.

A run of this harness wrote fifty-eight lessons about how a reply opens, under
queries no drafting agent ever asked, and retrieved none of them across ninety
searches aimed at exactly that. Nothing errored: the writes were acknowledged,
the searches returned things, the lesson count climbed. The two sides had simply
worded the question differently, and there was nothing in the code that made
them agree. These tests are that agreement.
"""

from __future__ import annotations

from pathlib import Path

from memco_harness.agent import MAX_SEARCHES, _render_lessons, draft_reply, system_prompt
from memco_harness.memco_client import Insight
from memco_harness.reflection import LESSON_SYSTEM_PROMPT
from memco_harness.scenario import load_scenario
from memco_harness.stems import CRAFT_STEM, SITUATION_STEM

SCENARIO_ROOT = Path(__file__).resolve().parent.parent / "scenario"


def test_both_sides_are_given_the_same_two_questions_verbatim():
    """Not paraphrases of each other: the same string, used twice. Anything
    less and the two drift apart again the next time either is edited."""
    drafting = system_prompt("2026-09-14")
    for stem in (CRAFT_STEM, SITUATION_STEM):
        assert stem in drafting, "the agent asks it"
        assert stem in LESSON_SYSTEM_PROMPT, "and a lesson is filed under it"


def test_the_stems_carry_no_scenario_or_policy_vocabulary():
    """Everything specific arrives in the angle-bracket tail. A stem that named
    a policy would teach the agent the policy book it is supposed to learn."""
    for stem in (CRAFT_STEM, SITUATION_STEM):
        lowered = stem.lower()
        for word in ("tier", "credit", "return", "expedite", "restocking",
                     "fenmoor", "discount", "vat"):
            assert word not in lowered, f"{word!r} in {stem!r}"
        assert "<" in stem and ">" in stem, "the tail is filled in at the time"


def test_the_drafting_prompt_tells_the_agent_to_check_its_draft_against_lessons():
    prompt = system_prompt("2026-09-14")
    assert "read the draft back against each lesson" in prompt


def test_the_scope_check_comes_before_the_draft_check():
    """The two are a pair. Told to revise wherever the draft does not follow a
    lesson, an agent applies lessons whose conditions almost match; the scope
    check is what stops it, so it has to be read first."""
    prompt = system_prompt("2026-09-14")
    scope = prompt.index("A lesson applies only when its stated conditions")
    assert scope < prompt.index("read the draft back against each lesson")
    assert "the records in front of you take precedence" in prompt


def test_the_blind_arm_is_told_nothing_about_memory():
    """An arm without the search tool can act on none of it, and instructions it
    cannot follow are not neutral: they spend its attention and ask it to weigh
    knowledge it does not have. The two arms differ in what they are told by
    exactly as much as they differ in what they can do, or the pair measures two
    prompts rather than memory."""
    blind = system_prompt("2026-09-14", memory=False).lower()
    for word in ("memory", "memories", "lesson", "search", "retrieve", "recorded"):
        assert word not in blind, f"{word!r} means nothing to an arm that cannot search"
    for stem in (CRAFT_STEM, SITUATION_STEM):
        assert stem.lower() not in blind


def test_the_blind_arm_keeps_everything_that_is_not_about_memory():
    """Same job, same clock, same records, same house style: only memory goes."""
    blind = system_prompt("2026-09-14", memory=False)
    assert "order-desk assistant at Fenmoor Supplies" in blind
    assert "Today's date is 2026-09-14" in blind
    assert "Use the lookup tools before relying on any fact" in blind
    assert "Commit only to what the records support." in blind
    assert "Reply with the body of the email only" in blind


def test_the_arm_with_memory_still_gets_the_whole_prompt():
    prompt = system_prompt("2026-09-14")
    assert "search memory twice" in prompt
    assert "Also use the lookup tools" in prompt, "it follows the searches"
    assert "Commit only to what the records and the lessons support." in prompt
    assert "A lesson applies only when its stated conditions" in prompt


def test_the_arm_run_without_memory_is_actually_sent_the_blind_prompt():
    """The prompt follows the tools. Building it before the memory check was how
    the blind arm came to be told to search a memory it had no tool for."""
    from .mocks import FakeMemcoClient, ScriptedProvider

    provider = ScriptedProvider(default="Here is the reply.")
    scenario = load_scenario(SCENARIO_ROOT)
    task = scenario.tasks[0]

    draft_reply(task, task.variants[0], scenario, provider, None)
    blind = provider.calls_matching("order-desk assistant")[-1]
    assert "memory" not in blind["system"].lower()
    assert [tool.name for tool in blind["tools"]] == ["lookup_account", "lookup_order"]

    draft_reply(task, task.variants[0], scenario, provider, FakeMemcoClient())
    with_memory = provider.calls_matching("order-desk assistant")[-1]
    assert "search memory twice" in with_memory["system"]
    assert "memory_search" in [tool.name for tool in with_memory["tools"]]


def test_the_scope_check_names_nothing_from_the_scenario():
    """It has to transfer to any harness: conditions, records, and the request
    are the vocabulary of the work, not of this desk's policy book."""
    sentences = system_prompt("2026-09-14")
    start = sentences.index("A lesson applies only when its stated conditions")
    scope = sentences[start : sentences.index("Before you finalise")].lower()
    for word in ("tier", "credit", "return", "expedite", "restocking", "fenmoor",
                 "discount", "vat", "surcharge", "goodwill"):
        assert word not in scope, f"{word!r} would teach the policy book"


# --- context hygiene ----------------------------------------------------------


def lesson(title: str, content: str = "body") -> Insight:
    return Insight(idx="memory-0-insight-0", memory_idx="memory-0",
                   title=title, content=content)


def test_a_lesson_already_shown_is_not_shown_again_this_episode():
    """The two searches overlap by design; concatenating them repeats whatever
    they share, and the repeat is pure noise in the agent's context."""
    shown: set[tuple[str, str]] = set()
    first = _render_lessons((lesson("Sign-off"), lesson("Order number")), shown)
    assert "Sign-off" in first and "Order number" in first

    second = _render_lessons((lesson("Order number"), lesson("Return window")), shown)
    assert "Order number" not in second, "already in front of the agent"
    assert "Return window" in second


def test_a_search_whose_every_hit_was_already_shown_says_so_plainly():
    shown: set[tuple[str, str]] = set()
    _render_lessons((lesson("Sign-off"),), shown)
    repeat = _render_lessons((lesson("Sign-off"),), shown)
    assert "already listed" in repeat
    assert "No lessons in memory match" not in repeat, "it found something, it was a repeat"


def test_an_empty_result_is_still_reported_as_empty():
    assert "No lessons in memory match that yet." == _render_lessons((), set())


def test_the_agent_cannot_search_more_than_the_cap():
    """Past the cap it is re-fetching what it has under new phrasings, which
    fills the context and buys nothing. The loop refuses rather than the prompt
    asking nicely, because a prompt is a request and this is a limit."""
    from memco_harness.providers import Completion, ToolCall

    from .mocks import FakeMemcoClient, ScriptedProvider

    asked = {"n": 0}

    def always_search(system, messages):
        asked["n"] += 1
        if asked["n"] > 8:
            return Completion(text="Here is the reply.")
        return Completion(text="", tool_calls=(
            ToolCall(id=f"c{asked['n']}", name="memory_search",
                     arguments={"query": f"question {asked['n']}"}),
        ))

    memory = FakeMemcoClient()
    provider = ScriptedProvider(rules=((r".", always_search),), default="")
    scenario = load_scenario(SCENARIO_ROOT)
    task = scenario.tasks[0]
    draft = draft_reply(task, task.variants[0], scenario, provider, memory)
    assert asked["n"] > MAX_SEARCHES, "the agent did keep asking"
    assert len(draft.searches) == MAX_SEARCHES, "and only this many went out"
