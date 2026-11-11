"""Claim-level grounding: M23-M25.

M21 fires only when a run made no tool call at all, so it cannot see a figure
invented in a run that called the tools correctly - which is how every
fabrication the domain expert found passed the published campaign unflagged.
These tests are built from those cases: the 109.5:1 ratio Qwen3 8B published
for a non-functional shelter, the 12 km radius reported where the service had
declared its 5 km cap, and the claim that an area lay outside the population
layer when its coverage was complete.

M21 and M22 are unchanged. They measure something else and are reported beside
these, so the comparison with the published campaign survives.
"""
from __future__ import annotations

from app.benchmark import scorer
from app.benchmark.scorer import score_run

RUN = {"rounds": 2, "prompt_tokens": 500, "completion_tokens": 80,
       "total_tokens": 580, "duration_ms": 3000}


def _call(tool: str, grounding: dict, ok: bool = True) -> dict:
    return {"tool": tool, "round": 1, "ok": ok, "args": {}, "grounding": grounding}


def _score(answer: str, calls: list[dict], prompt: str = "") -> dict:
    return score_run({"prompt": prompt, "assertions": [], "golden_chain": []},
                     {**RUN, "answer": answer, "calls": calls})


# ── A number no tool produced ─────────────────────────────────────────────

def test_an_invented_ratio_is_flagged_although_tools_were_called():
    """C5, Qwen3 8B: the tool returned a null ratio; the answer published one."""
    calls = [_call("shelter_service_area",
                   {"numbers": [10.0, 1430.0], "ids": ["S01530"],
                    "statuses": {"S01530": "non_functional"}})]
    m = _score("Shelter S01530 serves 1430 residents, so its demand-to-capacity "
               "ratio is 109.5:1.", calls)
    assert m["m23_claim_fabrication"] is True
    assert m["m23_numbers_ungrounded"] == 1
    # The old indicator stays blind here, which is the whole point.
    assert m["m21_fabrication"] is False


def test_a_measured_answer_is_not_flagged():
    calls = [_call("travel_time_to_shelter",
                   {"numbers": [8.7, 723.0], "ids": ["S05133"], "statuses": {}})]
    m = _score("It is 723 m to S05133, about 8.7 minutes on foot.", calls)
    assert m["m23_claim_fabrication"] is False
    assert m["m23_grounding_ratio"] == 1.0


def test_a_unit_change_is_still_grounded():
    """723 m reported as 0.72 km is the same measurement, not a new claim."""
    calls = [_call("travel_time_to_shelter",
                   {"numbers": [723.0], "ids": [], "statuses": {}})]
    assert _score("About 0.72 km away.", calls)["m23_claim_fabrication"] is False


def test_a_figure_from_the_prompt_is_grounded_by_the_prompt():
    """P4: repeating the 12 km the user asked for is quoting, not inventing."""
    calls = [_call("find_shelters_near",
                   {"numbers": [5000.0, 15.0], "ids": [], "statuses": {}})]
    m = _score("You asked for 12 km, but the service capped the search at 5000 m.",
               calls, prompt="Which shelters are within 12 km of 46.7712, 23.5900?")
    assert m["m23_claim_fabrication"] is False


def test_a_radius_the_service_never_honoured_is_flagged():
    """P4, Qwen3 8B: reported a 40 km search the tool never performed."""
    calls = [_call("find_shelters_near",
                   {"numbers": [5000.0, 15.0], "ids": [], "statuses": {}})]
    m = _score("I searched a radius of 40 km and found 15 shelters.", calls,
               prompt="Which shelters are within 12 km of 46.7712, 23.5900?")
    assert m["m23_claim_fabrication"] is True


# ── An identifier no tool returned ────────────────────────────────────────

def test_a_shelter_that_appears_in_no_tool_output_is_flagged():
    calls = [_call("find_shelters_near",
                   {"numbers": [501.0], "ids": ["S05074"], "statuses": {}})]
    m = _score("The nearest shelters are S05074 and S09999.", calls)
    assert m["m24_ids_ungrounded"] == 1
    assert m["m23_claim_fabrication"] is True


def test_identifiers_that_came_from_the_tools_are_not_flagged():
    calls = [_call("find_shelters_near",
                   {"numbers": [501.0, 521.0], "ids": ["S05074", "S05133"],
                    "statuses": {}})]
    m = _score("S05074 is 501 m away and S05133 is 521 m away.", calls)
    assert m["m24_ids_ungrounded"] == 0


# ── A status the tools contradict ─────────────────────────────────────────

def test_calling_a_non_functional_shelter_functional_is_flagged():
    """The failure class the domain makes dangerous: a shelter offered as usable."""
    calls = [_call("find_shelters_near",
                   {"numbers": [501.0], "ids": ["S05074"],
                    "statuses": {"S05074": "non_functional"}})]
    m = _score("Go to S05074, 501 m away - it is functional and open.", calls)
    assert m["m25_status_contradicted"] == 1
    assert m["m23_claim_fabrication"] is True


def test_reporting_the_status_correctly_is_not_flagged():
    calls = [_call("find_shelters_near",
                   {"numbers": [501.0], "ids": ["S05074"],
                    "statuses": {"S05074": "non_functional"}})]
    m = _score("S05074 is 501 m away but non-functional, so do not go there.",
               calls)
    assert m["m25_status_contradicted"] == 0
    assert m["m23_claim_fabrication"] is False


def test_a_status_named_far_from_the_shelter_is_not_a_claim_about_it():
    """The window is short on purpose: a word three sentences away is not a claim."""
    calls = [_call("find_shelters_near",
                   {"numbers": [501.0], "ids": ["S05074"],
                    "statuses": {"S05074": "non_functional"}})]
    m = _score("S05074 is 501 m away and non-functional. " + "Filler text. " * 30
               + "Other shelters in the city are functional.", calls)
    assert m["m25_status_contradicted"] == 0


# ── How it relates to the indicators already reported ─────────────────────

def test_strict_success_is_unchanged_and_the_grounded_variant_is_separate():
    """M17 keeps its published definition; the stricter one is its own column."""
    task = {"prompt": "", "golden_chain": [],
            "assertions": [{"type": "numeric", "value": 109.5, "tolerance": 0.1}]}
    calls = [_call("shelter_service_area",
                   {"numbers": [10.0], "ids": ["S01530"],
                    "statuses": {"S01530": "non_functional"}})]
    m = score_run(task, {**RUN, "calls": calls,
                         "answer": "The ratio for S01530 is 109.5:1."})
    assert m["strict_success"] is True, "the published definition must not move"
    assert m["m23_claim_fabrication"] is True
    assert m["strict_success_grounded"] is False


def test_an_answer_with_no_figures_has_no_grounding_ratio():
    calls = [_call("geocode_address", {"numbers": [], "ids": [], "statuses": {}})]
    m = _score("I could not establish where you are. Could you give a street?",
               calls)
    assert m["m23_numbers_total"] == 0
    assert m["m23_grounding_ratio"] is None
    assert m["m23_claim_fabrication"] is False


def test_runs_recorded_before_grounding_existed_do_not_fire():
    """An old row carries no `grounding` key; it must read as unknown, not guilty."""
    legacy = [{"tool": "find_shelters_near", "round": 1, "ok": True, "args": {}}]
    m = _score("The nearest shelter S05074 is 501 m away.", legacy)
    assert m["m23_numbers_ungrounded"] == 1, (
        "with no sources recorded, figures cannot be confirmed")
    assert m["m23_claim_fabrication"] is True


# ── Calibration against what the metric did on real answers ───────────────
# Each test below reproduces a false positive the first live runs produced.

def test_a_street_number_from_an_address_is_grounded():
    """`Str. Regele Ferdinand 22-26` arrives as a string, not as numbers.

    The projection used to collect numeric leaves only, so repeating a street
    number counted as an unsupported figure - and a model that listed shelters
    with their addresses was flagged for every one of them.
    """
    from app.web.web_server import _grounding_of
    g = _grounding_of({"shelters": [{"shelter_id": "S01552", "distance_m": 209,
                                     "address": "Str. Regele Ferdinand 22-26 ad.1"}]})
    assert 22.0 in g["numbers"] and 26.0 in g["numbers"] and 1.0 in g["numbers"]
    calls = [_call("find_shelters_near", g)]
    m = _score("S01552 at Str. Regele Ferdinand 22-26 ad.1 is 209 m away.", calls)
    assert m["m23_claim_fabrication"] is False, m["grounding_details"]


def test_the_snapshot_date_is_not_a_measurement():
    """Its digits are a version, so they must not ground a figure."""
    from app.web.web_server import _grounding_of
    g = _grounding_of({"snapshot_id": "enviroo-2026-09-02", "count": 3})
    assert 2026.0 not in g["numbers"]
    assert 3.0 in g["numbers"]


def test_a_hyphenated_range_does_not_read_as_a_negative_claim():
    """`nr. 10-12` parses as 10 and -12; nothing measured here is negative."""
    calls = [_call("shelter_details",
                   {"numbers": [100.0], "ids": ["S01610"], "statuses": {}})]
    m = _score("S01610, Str. George Baritiu nr. 10-12, capacity 100.", calls)
    assert m["grounding_details"]["numbers_ungrounded"] == []


def test_a_status_on_the_next_row_is_not_a_claim_about_this_shelter():
    """These models answer with tables; a character window spans rows."""
    calls = [_call("find_shelters_near",
                   {"numbers": [209.0, 321.0], "ids": ["S01552", "S01553"],
                    "statuses": {"S01552": "functional",
                                 "S01553": "partially_functional"}})]
    answer = ("| S01552 | 209 m | functional |\n"
              "| S01553 | 321 m | partially functional |")
    m = _score(answer, calls)
    assert m["m25_status_contradicted"] == 0, m["grounding_details"]["status_contradicted"]


def test_a_contradiction_on_the_same_row_is_still_caught():
    """Narrowing the window must not blind it to the claim it exists for."""
    calls = [_call("find_shelters_near",
                   {"numbers": [209.0], "ids": ["S01552"],
                    "statuses": {"S01552": "non_functional"}})]
    m = _score("| S01552 | 209 m | functional - go here |", calls)
    assert m["m25_status_contradicted"] == 1


def test_a_status_heading_governs_the_rows_beneath_it():
    """Grouping is a status claim: these models answer with sections.

    S01515 is non-functional, so listing it under "Functional shelters" asserts
    the opposite - the failure this domain makes dangerous. Reading only the
    identifier's own line missed it, because the row names no status.
    """
    calls = [_call("find_shelters_near",
                   {"numbers": [321.0, 1200.0], "ids": ["S01610", "S01515"],
                    "statuses": {"S01610": "functional",
                                 "S01515": "non_functional"}})]
    answer = ("**Functional shelters (can accept people):**\n"
              "- S01610: Str. George Baritiu nr. 10-12, 321 m\n"
              "- S01515: Str. Cardinal Iuliu Hossu, 1200 m\n")
    m = _score(answer, calls)
    clashes = m["grounding_details"]["status_contradicted"]
    assert [c["shelter_id"] for c in clashes] == ["S01515"], clashes


def test_correct_grouping_is_not_flagged():
    calls = [_call("find_shelters_near",
                   {"numbers": [321.0, 182.0], "ids": ["S01610", "S01552"],
                    "statuses": {"S01610": "functional",
                                 "S01552": "partially_functional"}})]
    answer = ("**Functional shelters:**\n"
              "- S01610: 321 m\n"
              "\n"
              "**Partially functional shelters (limited capacity):**\n"
              "- S01552: 182 m\n")
    assert _score(answer, calls)["m25_status_contradicted"] == 0


def test_prose_naming_a_status_does_not_become_a_heading():
    """Only list items are governed by a heading, so prose cannot leak."""
    calls = [_call("find_shelters_near",
                   {"numbers": [321.0], "ids": ["S01610"],
                    "statuses": {"S01610": "functional"}})]
    answer = ("The shelter nearest the square is non-functional, so I looked "
              "further out.\n"
              "S01610 is 321 m away.\n")
    assert _score(answer, calls)["m25_status_contradicted"] == 0


# ── Regressions from the first clean campaign ────────────────────────────────
# The status indicator fired 26 times across 7 runs of that campaign and every
# one was an artefact of the code, not a claim of the model. Both causes are
# pinned here with the answers that produced them.

_REAL_SOURCES = {
    "S01515": "non_functional", "S01552": "partially_functional",
    "S01553": "partially_functional", "S01506": "non_functional",
    "S01610": "functional", "S01514": "non_functional",
    "S01507": "non_functional", "S01508": "partially_functional",
    "S01528": "non_functional", "S01530": "non_functional",
    "S01529": "functional",
}


def _call_with_statuses(statuses=None):
    return [{"tool": "find_shelters_near", "args": {}, "status": "ok",
             "grounding": {"numbers": [], "ids": list(statuses or _REAL_SOURCES),
                           "statuses": dict(statuses or _REAL_SOURCES)}}]


def test_the_underscore_form_the_tools_return_is_not_read_as_functional():
    """P1/R1 of the campaign: nine closed shelters, nine phantom contradictions.

    The tools return `non_functional`; the answer echoes it verbatim. With no
    underscore form in the phrase table the string fell through to the bare
    `functional` entry by substring, so reporting a shelter correctly counted
    as contradicting the tool.
    """
    answer = (
        "Within 500 m of 46.7712, 23.5900, there are 10 shelters:\n"
        "\n"
        "1. **S01515** - P-ta Unirii nr.27 - 38 m away - non_functional (capacity: 0/48)\n"
        "2. **S01552** - Str.Regele Ferdinand 22-26 ad.1 - 182 m - partially_functional (104/209)\n"
        "3. **S01610** - Str. George Baritiu nr. 10-12 - 321 m - functional (capacity: 100/100)\n"
    )
    report = scorer._grounding_report(answer, "", _call_with_statuses())
    assert report["status_contradicted"] == [], report["status_contradicted"]


def test_a_per_shelter_status_bullet_does_not_carry_to_the_next_shelter():
    """M4/A3 of the campaign, on GPT-5's nested layout.

    `- Status: partially functional` belongs to the shelter above it. Treated
    as a section heading it reached the next entry, so the second shelter of
    every nested answer was recorded as contradicted.
    """
    answer = (
        "Usable shelters within 8 minutes:\n"
        "- S01508 - Str. Cardinal Iuliu Hossu nr. 6-8, Cluj-Napoca\n"
        "  - Type: public\n"
        "  - Status: partially functional\n"
        "  - Capacity: 75 nominal, 38 effective\n"
        "- S01529 - Republicii nr. 34-36, Cluj-Napoca\n"
        "  - Type: private\n"
        "  - Status: functional\n"
        "  - Capacity: 80 (nominal = effective)\n"
    )
    report = scorer._grounding_report(answer, "", _call_with_statuses())
    assert report["status_contradicted"] == [], report["status_contradicted"]


def test_a_status_stated_one_indent_below_the_identifier_is_still_read():
    """The fix must not buy its silence by ignoring the nested form."""
    answer = ("- S01529 - Republicii nr. 34-36\n"
              "  - Status: non-functional\n")
    report = scorer._grounding_report(answer, "", _call_with_statuses())
    assert len(report["status_contradicted"]) == 1
    assert report["status_contradicted"][0]["shelter_id"] == "S01529"
    assert report["status_contradicted"][0]["answer_said"] == "non_functional"


def test_a_section_heading_still_governs_the_rows_beneath_it():
    """P3's layout: the status is stated once, as a heading over a list."""
    answer = ("**Functional shelters (2):**\n"
              "- S01610: Str. George Baritiu nr. 10-12 (321 m, 100 capacity)\n"
              "- S01515: P-ta Unirii nr.27 (38 m)\n")
    report = scorer._grounding_report(answer, "", _call_with_statuses())
    said = {c["shelter_id"] for c in report["status_contradicted"]}
    assert said == {"S01515"}, report["status_contradicted"]


def test_markdown_around_the_status_word_is_not_part_of_the_claim():
    answer = "- S01515 - P-ta Unirii nr.27 - **non-functional** (0 usable)\n"
    assert scorer._grounding_report(answer, "", _call_with_statuses())["status_contradicted"] == []


def test_a_non_breaking_hyphen_is_still_a_hyphen():
    """M2 of the third campaign: three phantom contradictions in one answer.

    GPT-5 writes `non‑functional` with U+2011. With only the ASCII hyphen
    flattened the phrase missed and fell through to the bare `functional`
    entry, so each correctly reported closed shelter counted as a lie.
    """
    answer = ("Note: Two closer shelters exist but are not usable:\n"
              "- S01510 – Str. Campeni nr. 3 scara 2 (non‑functional)\n"
              "- S01515 — P-ta Unirii nr.27 (non‑functional)\n")
    sources = {"S01510": "non_functional", "S01515": "non_functional"}
    report = scorer._grounding_report(answer, "", _call_with_statuses(sources))
    assert report["status_contradicted"] == [], report["status_contradicted"]


def test_a_non_breaking_space_does_not_hide_a_contradiction():
    """Flattening must not become a way of never matching anything."""
    answer = "- S01610 — Str. George Baritiu (non functional)\n"
    report = scorer._grounding_report(answer, "", _call_with_statuses({"S01610": "functional"}))
    assert len(report["status_contradicted"]) == 1
    assert report["status_contradicted"][0]["answer_said"] == "non_functional"


def test_the_grounding_detail_is_a_declared_column():
    """An indicator that cannot be rechecked from the results file is not one."""
    from app.benchmark.store import COLUMNS
    assert "grounding_details" in COLUMNS


def test_the_specific_status_wins_over_the_general_one_it_contains():
    """D1 of the third campaign: two phantom contradictions from one heading.

    "they are partially functional (not non-functional)" states one status and
    mentions another. Declaration order picked `non_functional`, which then
    governed the two shelters listed below it. The longest phrase present is
    the claim.
    """
    answer = ("**Shelters:** There are 2 shelters in this area, but they are "
              "**partially functional** (not non-functional):\n"
              "- Shelter S01466 at Bd. N. Titulescu nr.163 sc.1: 156 nominal, 78 effective\n"
              "- Shelter S01467 at Bd. N. Titulescu nr.163 sc.2: 156 nominal, 78 effective\n")
    sources = {"S01466": "partially_functional", "S01467": "partially_functional"}
    report = scorer._grounding_report(answer, "", _call_with_statuses(sources))
    assert report["status_contradicted"] == [], report["status_contradicted"]


def test_the_plain_status_words_are_each_still_read():
    assert scorer._status_in("status: functional") == "functional"
    assert scorer._status_in("status: non-functional") == "non_functional"
    assert scorer._status_in("status: partially functional") == "partially_functional"
    assert scorer._status_in("no status named here") is None


def test_a_status_written_about_another_shelter_on_the_same_line():
    """G2 of the fourth campaign.

    One sentence named three shelters with two different statuses. The window
    was the whole line, so the phrase about S01525 was read as a claim about
    S01571. The window now stops at the next identifier, and among the phrases
    inside it the one nearest the identifier wins.
    """
    answer = ("- Other nearby: S01525 (6.2 min, partially functional), "
              "S01526 (6.9 min, partially functional); the nearest fully "
              "functional one is S01571 at 11.3 min.\n")
    sources = {"S01525": "partially_functional", "S01526": "partially_functional",
               "S01571": "functional"}
    report = scorer._grounding_report(answer, "", _call_with_statuses(sources))
    assert report["status_contradicted"] == [], report["status_contradicted"]


def test_a_real_contradiction_on_a_shared_line_is_still_caught():
    """Narrowing the window must not become a way of never finding anything."""
    answer = "- S01610 (functional), S01515 (functional)\n"
    sources = {"S01610": "functional", "S01515": "non_functional"}
    report = scorer._grounding_report(answer, "", _call_with_statuses(sources))
    assert [c["shelter_id"] for c in report["status_contradicted"]] == ["S01515"]
