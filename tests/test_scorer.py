"""
The grader's reading of an answer, and of the arguments a model supplied.

These are not unit tests of convenience: each one pins a defect that produced
wrong numbers in results.csv before it was found. A grader that misreads is
worse than no grader, because its output looks like a measurement.
"""
from __future__ import annotations

import pytest

from app.benchmark.scorer import (
    _address_tokens,
    _addresses_equal,
    _compare_args,
    _eval_assertion,
    _numbers_in,
    score_run,
)


def _values(text: str) -> list[float]:
    return [v for v, _ in _numbers_in(text)]


# ── What counts as a number in an answer ──────────────────────────────────

def test_thousands_separator_is_one_number_not_a_fraction():
    """`25,792` residents was being read as 25.792, which is how a population
    of twenty-five thousand passed an assertion written for tens."""
    assert _values("About 25,792 residents.") == [25792.0]
    assert _values("A total of 1,234,567 people.") == [1234567.0]
    assert _values("Population 23 801 in the area.") == [23801.0]


def test_decimal_comma_survives_the_thousands_rule():
    """Three digits after the separator means thousands; one or two keep the
    decimal comma a model writing Romanian uses."""
    assert _values("Durata este de 2,8 minute.") == [2.8]
    assert _values("Aria este 1,03 km patrati.") == [1.03]


def test_shelter_identifiers_are_not_measurements():
    """`S01545` was contributing 1545 to the numbers of an answer, which let a
    run that measured nothing satisfy an assertion by naming a shelter."""
    assert _values("The nearest is S01545, 612 m away.") == [612.0]


def test_house_numbers_are_not_measurements():
    """The failure this was written for: `Str. Garii nr. 19` satisfied an
    interval of 8 to 20 minutes on an answer that reported no walking time."""
    assert _values("S01545 at Str. Garii nr. 19.") == []
    assert _values("Str. Garii nr. 19; the walk takes 11.9 minutes.") == [11.9]

    interval = {"type": "interval", "min": 8, "max": 20}
    assert _eval_assertion(interval, "The shelter is at Str. Garii nr. 19.", [], 1) is False


def test_coordinates_are_read_as_two_numbers():
    assert _values("Coordinates 46.7712, 23.5900.") == [46.7712, 23.59]


# ── Addresses ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("written", [
    "Strada Horea 27, Cluj-Napoca",
    "Str. Horea nr. 27, Cluj-Napoca, Romania",
    "Horea 27, Cluj",
    "Cluj-Napoca, Strada Horea 27",
])
def test_the_same_address_written_four_ways_is_one_address(written):
    assert _addresses_equal("Strada Horea 27, Cluj-Napoca", written)


def test_extra_detail_in_an_address_is_a_different_query():
    """A1's whole subject: block, staircase and floor locate a flat inside a
    building, a geocoder is worse for having them, and sending them has to be
    visible as a parameter error rather than smoothed away."""
    assert not _addresses_equal(
        "Strada Horea 27, Cluj-Napoca",
        "Strada Horea 27, bloc 4, scara 2, etaj 3, Cluj-Napoca")


def test_a_different_street_is_a_different_address():
    assert not _addresses_equal("Strada Horea 27, Cluj-Napoca",
                                "Strada Napoca 27, Cluj-Napoca")


def test_address_tokens_drop_only_the_words_that_carry_nothing():
    assert _address_tokens("Str. Horea nr. 27, Cluj-Napoca") == {"horea", "27"}


def test_naming_the_city_or_omitting_it_is_the_same_address():
    """`cluj` is a stopword for the same reason `napoca` always was.

    With only half the city name ignored, adding or dropping it flipped the
    comparison: a model that helpfully appended `Cluj-Napoca` to an address the
    chain stored without it was recorded as having passed a wrong value. Which
    place the geocoder actually returned is a question about the result, and the
    task's manual assertion is what asks it.
    """
    for a, b in (("Strada Horea 27", "Strada Horea 27, Cluj-Napoca"),
                 ("Strada Horea 27, Cluj-Napoca", "Str. Horea nr. 27"),
                 ("Piata Unirii, Cluj-Napoca", "Piata Unirii Cluj")):
        assert _addresses_equal(a, b), (a, b)


def test_query_is_scored_on_content_not_on_form():
    golden = {"query": "Strada Horea 27, Cluj-Napoca"}
    result = _compare_args(golden, {"query": "Str. Horea nr. 27, Cluj"})
    assert result["corecti"] == 1 and not result["erori"]


# ── Arguments ─────────────────────────────────────────────────────────────

def test_a_coordinate_sent_with_a_decimal_comma_is_an_error_of_its_own():
    """`46,7712` is refused by the tool. The grader used to normalize the comma
    and score it correct, hiding the failure P1 exists to provoke."""
    result = _compare_args({"lat": 46.7712, "lon": 23.59},
                           {"lat": "46,7712", "lon": 23.59})
    assert result["corecti"] == 1
    assert [e["type"] for e in result["erori"]] == ["comma_decimal"]


def test_a_coordinate_sent_as_a_dotted_string_is_still_correct():
    result = _compare_args({"lat": 46.7712}, {"lat": "46.7712"})
    assert result["corecti"] == 1 and not result["erori"]


def test_swapped_coordinates_keep_their_own_error_type():
    result = _compare_args({"lat": 46.7712, "lon": 23.59},
                           {"lat": 23.59, "lon": 46.7712})
    assert any(e["type"] == "latlon_swap" for e in result["erori"])


# ── Assertions ────────────────────────────────────────────────────────────

_TWO_BRANCHES = ["geocode_address", "find_shelters_near", "travel_time_to_shelter",
                 "geocode_address", "find_shelters_near", "travel_time_to_shelter"]


def test_two_independent_branches_may_be_grouped_or_interleaved():
    """G2 compares two addresses. Running both geocodes and then both routes is
    a plan, not a mistake; the assertion failed all nine runs over it."""
    assertion = {"type": "tool_sequence", "order": "any", "value": _TWO_BRANCHES}
    grouped = ["geocode_address", "geocode_address", "find_shelters_near",
               "find_shelters_near", "travel_time_to_shelter", "travel_time_to_shelter"]
    assert _eval_assertion(assertion, "", _TWO_BRANCHES, 6) is True
    assert _eval_assertion(assertion, "", grouped, 6) is True


def test_order_any_still_requires_every_call():
    assertion = {"type": "tool_sequence", "order": "any", "value": _TWO_BRANCHES}
    one_branch = ["geocode_address", "find_shelters_near", "travel_time_to_shelter"]
    assert _eval_assertion(assertion, "", one_branch, 3) is False


def test_ordered_sequences_stay_ordered_by_default():
    assertion = {"type": "tool_sequence",
                 "value": ["geocode_address", "find_shelters_near"]}
    assert _eval_assertion(assertion, "", ["find_shelters_near", "geocode_address"], 2) is False


# ── Reporting a refusal is not concealing a failure ───────────────────────

@pytest.mark.parametrize("answer", [
    "The address is ambiguous - several places match. Which city do you mean?",
    "Cluj-Napoca does not have a metro, so there is no metro station to measure from.",
    "Ventilation is not recorded in this inventory.",
    "Adresa este ambigua, sunt mai multe rezultate.",
])
def test_a_reported_refusal_does_not_count_as_a_hidden_failure(answer):
    """M22 asks whether a model hid a failed tool call. A model that passes the
    refusal on has hidden nothing - and C2, C3 and C4 were being flagged for
    doing exactly that."""
    task = {"assertions": [{"type": "max_calls", "value": 4}]}
    run = {"calls": [{"tool": "geocode_address", "args": {}, "ok": False,
                      "error": "geocoding_ambiguous", "round": 1}],
           "answer": answer, "rounds": 1, "prompt_tokens": 500,
           "completion_tokens": 80, "total_tokens": 580, "duration_ms": 3000}
    assert score_run(task, run)["m22_hallucination"] is False


def test_an_unreported_failure_is_still_caught():
    task = {"assertions": [{"type": "max_calls", "value": 4}]}
    run = {"calls": [{"tool": "geocode_address", "args": {}, "ok": False,
                      "error": "geocoding_ambiguous", "round": 1}],
           "answer": "The nearest shelter is 320 m away, about 4 minutes on foot.",
           "rounds": 1, "prompt_tokens": 500, "completion_tokens": 80,
           "total_tokens": 580, "duration_ms": 3000}
    assert score_run(task, run)["m22_hallucination"] is True

# -- Artefacts of the chain, not mistakes of the model ---------------------
# Each test below pins a rule that recorded a defensible run as an error. They
# are regression tests in the strict sense: every one of them failed before the
# corresponding correction, on values taken from the campaign's own results.

def _route(shelter_id: str, ok: bool = True) -> dict:
    return {"tool": "travel_time_to_shelter", "round": 1, "ok": ok,
            "args": {"lat": 46.7712, "lon": 23.59, "shelter_id": shelter_id}}


def _b3_chain() -> list[dict]:
    """B3's reference chain: find the candidates, then route both tied ones."""
    return [
        {"tool": "find_shelters_near", "args": {"lat": 46.7712, "lon": 23.59}},
        {"tool": "travel_time_to_shelter",
         "args": {"lat": 46.7712, "lon": 23.59, "shelter_id": "S01552"}},
        {"tool": "travel_time_to_shelter",
         "args": {"lat": 46.7712, "lon": 23.59, "shelter_id": "S01553"}},
    ]


def test_two_tied_shelters_may_be_routed_in_either_order():
    """B3: the chain lists S01552 first, but the task does not ask for an order.

    Routing the pair the other way round produced two wrong-value errors - the
    whole of the B3 artefact in the published campaign, and two of the five
    argument errors recorded in the campaign that followed it.
    """
    calls = [{"tool": "find_shelters_near", "round": 1, "ok": True,
              "args": {"lat": 46.7712, "lon": 23.59}},
             _route("S01553"), _route("S01552")]
    run = {"calls": calls, "answer": "Both are 2.8 minutes away: a tie.",
           "rounds": 2, "prompt_tokens": 500, "completion_tokens": 80,
           "total_tokens": 580, "duration_ms": 3000}
    m = score_run({"golden_chain": _b3_chain(), "assertions": []}, run)
    assert m["param_errors"] == [], m["param_errors"]
    assert m["m4_param_accuracy"] == 1.0
    assert m["calls_useful"] == 3 and m["calls_redundant"] == 0


def test_the_canonical_order_still_scores_clean():
    """The correction must not be a blanket amnesty: the listed order is fine."""
    calls = [{"tool": "find_shelters_near", "round": 1, "ok": True,
              "args": {"lat": 46.7712, "lon": 23.59}},
             _route("S01552"), _route("S01553")]
    run = {"calls": calls, "answer": "A tie at 2.8 minutes.", "rounds": 2,
           "prompt_tokens": 500, "completion_tokens": 80, "total_tokens": 580,
           "duration_ms": 3000}
    m = score_run({"golden_chain": _b3_chain(), "assertions": []}, run)
    assert m["param_errors"] == [] and m["m4_param_accuracy"] == 1.0


def test_a_genuinely_wrong_shelter_is_still_an_error():
    """Reordering is forgiven; routing a shelter the chain never named is not."""
    calls = [{"tool": "find_shelters_near", "round": 1, "ok": True,
              "args": {"lat": 46.7712, "lon": 23.59}},
             _route("S01552"), _route("S09999")]
    run = {"calls": calls, "answer": "One is faster.", "rounds": 2,
           "prompt_tokens": 500, "completion_tokens": 80, "total_tokens": 580,
           "duration_ms": 3000}
    m = score_run({"golden_chain": _b3_chain(), "assertions": []}, run)
    errors = [e for e in m["param_errors"] if e.get("param") == "shelter_id"]
    assert len(errors) == 1, m["param_errors"]
    assert errors[0]["received"] == "S09999"


def test_the_failed_attempt_does_not_win_the_pairing():
    """A refused call then a correct retry is scored on the retry, as before."""
    calls = [{"tool": "find_shelters_near", "round": 1, "ok": True,
              "args": {"lat": 46.7712, "lon": 23.59}},
             _route("S01552", ok=False), _route("S01553"), _route("S01552")]
    run = {"calls": calls, "answer": "A tie.", "rounds": 3, "prompt_tokens": 500,
           "completion_tokens": 80, "total_tokens": 580, "duration_ms": 3000}
    m = score_run({"golden_chain": _b3_chain(), "assertions": []}, run)
    assert m["param_errors"] == [], m["param_errors"]


def test_independent_branches_may_be_grouped():
    """G1 and G2: `order: any` counts the calls, and depends_on keeps the order.

    Three runs of the published campaign were marked down for geocoding both
    addresses before routing either - a valid plan for two independent branches.
    """
    grouped = ["geocode_address", "geocode_address", "find_shelters_near",
               "find_shelters_near", "travel_time_to_shelter",
               "travel_time_to_shelter"]
    strict = {"type": "tool_sequence",
              "value": ["geocode_address", "find_shelters_near",
                        "travel_time_to_shelter", "geocode_address",
                        "find_shelters_near", "travel_time_to_shelter"]}
    assert _eval_assertion(strict, "", grouped, len(grouped)) is False
    assert _eval_assertion({**strict, "order": "any"}, "", grouped,
                           len(grouped)) is True


def test_order_any_still_requires_every_call_to_happen():
    """It relaxes the order, not the obligation: a missing step still fails."""
    short = ["geocode_address", "find_shelters_near", "travel_time_to_shelter",
             "geocode_address", "find_shelters_near"]
    a = {"type": "tool_sequence", "order": "any",
         "value": ["geocode_address", "find_shelters_near",
                   "travel_time_to_shelter", "geocode_address",
                   "find_shelters_near", "travel_time_to_shelter"]}
    assert _eval_assertion(a, "", short, len(short)) is False


def test_equals_is_no_longer_silently_a_substring_test():
    """It promised equality and tested inclusion; now it asks for the expert."""
    a = {"type": "equals", "value": "2.8 minutes"}
    assert _eval_assertion(a, "About 2.8 minutes on foot.", [], 0) is None
