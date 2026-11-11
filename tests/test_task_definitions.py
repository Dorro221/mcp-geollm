"""The task definitions, read as the experimental apparatus they are.

A reference chain or an assertion may only require what the task's own prompt
requires. Where it required more, a defensible run was recorded as a model
error: seventeen of the thirty argument errors in the first campaign, and all
five of those in the campaign that followed it. Each test here pins one such
correction to the file, so that a future edit cannot quietly undo it.
"""
from __future__ import annotations

import json
import os

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_GOLDEN = os.path.join(_ROOT, "benchmark", "golden_chains.json")


@pytest.fixture(scope="module")
def tasks() -> dict:
    with open(_GOLDEN, encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("tasks", data)
    return {t["id"]: t for t in raw} if isinstance(raw, list) else raw


def _phrase_lexicon(task: dict) -> list[str]:
    a = next(x for x in task["assertions"] if x["type"] == "contains_any")
    return [v.lower() for v in a["value"]]


def _sequences(task: dict) -> list[dict]:
    return [a for a in task["assertions"] if a["type"] == "tool_sequence"]


# -- Reference chains may not fix what the prompt leaves open --------------

def test_b2_does_not_pin_a_radius_the_prompt_never_states(tasks):
    """B2 asks for the nearest shelter, naming no radius.

    The chain had pinned 200 m, so both commercial models were recorded as
    having passed a wrong value for using the documented 800 m default.
    """
    prompt = tasks["B2"]["prompt"].lower()
    assert "200" not in prompt, "if the prompt now states a radius, re-pin it"
    searches = [s for s in tasks["B2"]["golden_chain"]
                if s["tool"] == "find_shelters_near"]
    assert searches, "B2 must still search"
    for step in searches:
        assert "radius_m" not in step.get("args", {}), (
            "B2 pins a radius again; the prompt does not state one")


def test_a1_accepts_the_building_metadata_in_either_language(tasks):
    """A1's prompt is English, its address Romanian; both renderings are valid.

    The accepted variant existed only as `block/staircase/floor`, so a model
    that carried the detail over as `bloc/scara/etaj` was scored wrong.
    """
    step = tasks["A1"]["golden_chain"][0]
    variants = " | ".join(step["arg_alternatives"]["query"]).lower()
    assert "staircase" in variants, "the English rendering must stay accepted"
    assert "scara" in variants, "the Romanian rendering must be accepted too"


# -- Independent steps may run in any order -------------------------------

@pytest.mark.parametrize("task_id", ["G1", "G2"])
def test_independent_branches_are_not_scored_on_their_order(tasks, task_id):
    """Both tasks hold steps that depend on a common parent, not on each other.

    G1 measures capacity and population over one polygon; G2 runs the same
    chain on two addresses. Grouping the independent calls is a valid plan, and
    the ordering that does matter is carried by `depends_on`.
    """
    for seq in _sequences(tasks[task_id]):
        assert seq.get("order") == "any", (
            f"{task_id} scores independent calls on their order again")


def test_tasks_with_a_single_branch_keep_their_strict_order(tasks):
    """The relaxation is specific: a genuinely sequential chain stays strict."""
    for task_id in ("A1", "B2", "D1"):
        for seq in _sequences(tasks[task_id]):
            assert seq.get("order") != "any", (
                f"{task_id} is a single dependent chain; its order is the test")


# -- A phrase lexicon may not accept a different claim --------------------

def test_c5_requires_the_ratio_to_be_called_undefined(tasks):
    """Naming the status is not the claim under test.

    With `non-functional` in the lexicon, an answer that named the status and
    then published an invented ratio of 109.5:1 passed the assertion.
    """
    lex = _phrase_lexicon(tasks["C5"])
    assert not any("functional" in v for v in lex), (
        "C5 accepts a status phrase again, which an invented ratio can satisfy")
    assert any("defined" in v or "computed" in v or "calculated" in v
               for v in lex), "C5 must still accept a correct report"


def test_p4_requires_the_cap_value_not_merely_the_word_limit(tasks):
    """`limit` passed on "within the 12 km limit" - the failure P4 exists for."""
    lex = _phrase_lexicon(tasks["P4"])
    for vague in ("maximum", "capped", "limit", "limita", "maxim"):
        assert vague not in lex, f"P4 accepts the bare word {vague!r} again"
    assert any("5" in v for v in lex), "P4 must still accept the 5 km cap"


def test_d1_accepts_more_than_one_wording_of_a_missing_field(tasks):
    """Declaring the gap is the competence; the phrasing is not."""
    lex = _phrase_lexicon(tasks["D1"])
    assert len(lex) >= 12, "D1's lexicon has been narrowed again"
    for wording in ("not recorded", "not tracked", "no information"):
        assert wording in lex, f"D1 no longer accepts {wording!r}"


# -- Shape of the file ----------------------------------------------------

def test_only_r1_is_decided_mechanically(tasks):
    """The mechanical checks screen runs; they do not certify answers.

    Nineteen of the twenty tasks carry a manual assertion, so an expert reads
    what the answer means. R1 is the exception - a reverse geocode whose output
    is fully determined - and it is pinned here so that a task losing its manual
    assertion shows up as a failure rather than as a silently cheaper campaign.
    """
    without = {tid for tid, task in tasks.items()
               if "manual" not in {a["type"] for a in task.get("assertions") or []}}
    assert without == {"R1"}, (
        f"tasks decided mechanically changed: {sorted(without)}")


def test_the_campaign_still_has_twenty_tasks(tasks):
    assert len(tasks) == 20
