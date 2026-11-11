"""Two readers of the same run, which section 6.5 could not report.

An observation used to be keyed by run_id alone, so a second annotator's
verdict replaced the first in both files: agreement was not merely unrecorded,
it was destroyed on write. The key is now (run_id, evaluator).
"""
from __future__ import annotations

import csv
import io
import os

import pytest

from app import config
from app.benchmark import store


@pytest.fixture
def results(tmp_path):
    """A results dir of its own, so the campaign file is never touched."""
    before = (store.RESULTS_DIR, store.RESULTS_CSV, store.OBSERVATIONS_CSV,
              config.EVALUATOR, config.EVALUATOR_PRIMARY)
    store.RESULTS_DIR = str(tmp_path)
    store.RESULTS_CSV = os.path.join(str(tmp_path), "metrics.csv")
    store.OBSERVATIONS_CSV = os.path.join(str(tmp_path), "observatii.csv")
    config.EVALUATOR_PRIMARY = "E1"
    store.append_result({"ts": store.now_iso(), "run_id": "C5__gpt-5__r1",
                         "task_id": "C5", "model_id": "gpt-5", "repetition": 1,
                         "descriptions_regime": "rich"})
    yield
    (store.RESULTS_DIR, store.RESULTS_CSV, store.OBSERVATIONS_CSV,
     config.EVALUATOR, config.EVALUATOR_PRIMARY) = before


def _rows(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_two_evaluators_of_one_run_both_survive(results):
    store.append_observation("C5__gpt-5__r1", "correct", "well grounded", "E1")
    store.append_observation("C5__gpt-5__r1", "partial", "missed the caveat", "E2")
    obs = _rows(store.OBSERVATIONS_CSV)
    assert len(obs) == 2, "the second reading replaced the first"
    assert {r["evaluator"] for r in obs} == {"E1", "E2"}
    assert {r["manual_verdict"] for r in obs} == {"correct", "partial"}


def test_the_same_evaluator_revising_a_verdict_still_overwrites(results):
    """Revisiting one's own reading was the behaviour worth keeping."""
    store.append_observation("C5__gpt-5__r1", "correct", "first pass", "E1")
    store.append_observation("C5__gpt-5__r1", "partial", "read it again", "E1")
    obs = _rows(store.OBSERVATIONS_CSV)
    assert len(obs) == 1
    assert obs[0]["manual_verdict"] == "partial"
    assert obs[0]["note"] == "read it again"


def test_each_evaluator_writes_their_own_column_in_metrics(results):
    store.set_manual_verdict("C5__gpt-5__r1", "correct", evaluator="E1")
    store.set_manual_verdict("C5__gpt-5__r1", "partial", evaluator="E2")
    row = _rows(store.RESULTS_CSV)[0]
    assert row["manual_verdict"] == "correct"
    assert row["manual_verdict_2"] == "partial"


def test_a_disagreement_is_visible_in_one_row(results):
    """What the agreement statistic is computed from."""
    store.set_manual_verdict("C5__gpt-5__r1", "correct", evaluator="E1")
    store.set_manual_verdict("C5__gpt-5__r1", "incorrect", evaluator="E2")
    row = _rows(store.RESULTS_CSV)[0]
    assert row["manual_verdict"] != row["manual_verdict_2"]


def test_single_rater_use_is_unchanged(results):
    """With one reader and no identity passed, nothing about the old flow moves."""
    config.EVALUATOR = "E1"
    store.append_observation("C5__gpt-5__r1", "correct", "note")
    store.set_manual_verdict("C5__gpt-5__r1", "correct")
    assert _rows(store.RESULTS_CSV)[0]["manual_verdict"] == "correct"
    assert _rows(store.RESULTS_CSV)[0]["manual_verdict_2"] == ""
    assert len(_rows(store.OBSERVATIONS_CSV)) == 1


def test_both_verdict_columns_are_declared():
    assert "manual_verdict" in store.COLUMNS
    assert "manual_verdict_2" in store.COLUMNS
    assert "evaluator" in store.OBSERVATION_COLUMNS
