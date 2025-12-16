import pytest
from datasets import Dataset

from euroeval.metrics import (
    accuracy_ambig_metric,
    accuracy_disambig_metric,
    bias_adjusted_accuracy_ambig_metric,
    bias_adjusted_accuracy_disambig_metric,
    bias_ambig_metric,
    bias_disambig_metric,
)


@pytest.fixture
def make_dataset():
    """
    Helper that builds a small HuggingFace Dataset with the required columns:
      - context_condition: "ambig" or "disambig"
      - stereo_idx, counter_idx, unknown_idx, correct_idx
    """
    def _make(cond, indices, correct, n):
        stereo, counter, unknown = indices
        records = []
        for _ in range(n):
            records.append({
                "context_condition": cond,
                "stereo_idx": stereo,
                "counter_idx": counter,
                "unknown_idx": unknown,
                "correct_idx": correct,
            })
        return Dataset.from_list(records)
    return _make


# --- BiasA tests (ambiguous contexts) ---

def test_biasa_all_unknowns(make_dataset):
    ds    = make_dataset("ambig", [1, 2, 0], None, n=3)
    preds = [0, 0, 0]
    # references arg is ignored by the metric, so we can pass an empty list
    assert bias_ambig_metric(preds, [], ds, None, None) == pytest.approx(0.0)

def test_biasa_stereo_and_counter(make_dataset):
    ds    = make_dataset("ambig", [1, 2, 0], None, n=5)
    preds = [1, 1, 1, 2, 2]  # three stereotype picks, two counter
    assert bias_ambig_metric(preds, [], ds, None, None) == pytest.approx((3-2)/5)

def test_biasa_ignores_undetectable(make_dataset):
    # one prediction not in label mapping => discarded,
    # but still counts toward n_ambiguous
    ds    = make_dataset("ambig", [1, 2, 0], None, n=3)
    preds = [1, "X", 2]
    # stereo=1, counter=2 => (1-1)/3 = 0.0
    assert bias_ambig_metric(preds, [], ds, None, None) == pytest.approx(0.0)

# --- AccuracyA tests (ambiguous contexts) ---

def test_accuracy_ambig_all_unknown(make_dataset):
    ds    = make_dataset("ambig", [1, 2, 0], None, n=4)
    preds = [0, 0, 0, 0]  # all correctly 'unknown'
    assert accuracy_ambig_metric(preds, [], ds, None, None) == pytest.approx(1.0)

def test_accuracy_ambig_mixed(make_dataset):
    ds    = make_dataset("ambig", [1, 2, 0], None, n=4)
    preds = [0, 1, 0, 2]  # two correct 'unknown', two wrong
    assert accuracy_ambig_metric(preds, [], ds, None, None) == pytest.approx(0.5)

# --- BiasD tests (disambiguated contexts) ---

def test_biasd_no_disambig(make_dataset):
    # dataset contains only ambiguous contexts → no disambig → None
    ds1 = make_dataset("ambig", [1, 2, 0], 1, n=1)
    assert bias_disambig_metric([], [], ds1, None, None) is None
    # even if we pass a prediction, still None
    assert bias_disambig_metric([1], [], ds1, None, None) is None

def test_biasd_all_wrong_or_unknown(make_dataset):
    ds    = make_dataset("disambig", [1, 2, 0], 1, n=4)
    preds = [0,2,0,2]  # all wrong or unknown
    assert bias_disambig_metric(preds, [], ds, None, None) == pytest.approx(0.0)

def test_biasd_counts_correct(make_dataset):
    # 2 biased contexts (gold=1), 2 counter contexts (gold=2)
    ds    = make_dataset("disambig", [1, 2, 0], None, n=0)
    # we need mixed gold labels, so build manually:
    recs = [
        {"context_condition":"disambig","stereo_idx":1,"counter_idx":2,"unknown_idx":0,"correct_idx":1},
        {"context_condition":"disambig","stereo_idx":1,"counter_idx":2,"unknown_idx":0,"correct_idx":1},
        {"context_condition":"disambig","stereo_idx":1,"counter_idx":2,"unknown_idx":0,"correct_idx":2},
        {"context_condition":"disambig","stereo_idx":1,"counter_idx":2,"unknown_idx":0,"correct_idx":2},
    ]
    ds = Dataset.from_list(recs)
    # preds: correct in first two, correct in third, wrong in fourth
    preds = [1,1,2,0]
    # (2 correct_biased − 1 correct_counter)/4 = 0.25
    assert bias_disambig_metric(preds, [], ds, None, None) == pytest.approx(0.25)

# --- AccuracyD tests (disambiguated contexts) ---

def test_accuracy_disambig_all_correct(make_dataset):
    ds    = make_dataset("disambig", [1, 2, 0], 1, n=3)
    preds = [1, 1, 1]
    assert accuracy_disambig_metric(preds, [], ds, None, None) == pytest.approx(1.0)

def test_accuracy_disambig_half_correct(make_dataset):
    ds    = make_dataset("disambig", [1, 2, 0], 1, n=4)
    preds = [1, 2, 1, 2]  # two correct, two wrong
    assert accuracy_disambig_metric(preds, [], ds, None, None) == pytest.approx(0.5)

# --- Bias-adjusted accuracy metrics ---

def test_bias_adjusted_accuracy_ambig_zero_at_chance(make_dataset):
    ds = make_dataset("ambig", [1, 2, 0], None, n=3)
    preds = [0, 1, 2]  # 1 correct unknown, one stereo, one counter => acc = 1/3, bias = 0
    assert bias_adjusted_accuracy_ambig_metric(preds, [], ds, None, None) == pytest.approx(1/3)


def test_bias_adjusted_accuracy_ambig_above_chance_low_bias(make_dataset):
    ds = make_dataset("ambig", [1, 2, 0], None, n=4)
    preds = [0, 0, 1, 2]  # acc = 0.5, bias = 0
    assert bias_adjusted_accuracy_ambig_metric(preds, [], ds, None, None) == pytest.approx(0.5)


def test_bias_adjusted_accuracy_disambig_below_chance(make_dataset):
    ds = make_dataset("disambig", [1, 2, 0], 1, n=3)
    preds = [2, 2, 2]  # acc = 0 (below chance) => overall should be 0
    assert bias_adjusted_accuracy_disambig_metric(preds, [], ds, None, None) == pytest.approx(0.0)


def test_bias_adjusted_accuracy_disambig_penalizes_bias(make_dataset):
    # 2 stereotype contexts (correct=1), 2 counter contexts (correct=2)
    recs = [
        {"context_condition": "disambig", "stereo_idx": 1, "counter_idx": 2, "unknown_idx": 0, "correct_idx": 1},
        {"context_condition": "disambig", "stereo_idx": 1, "counter_idx": 2, "unknown_idx": 0, "correct_idx": 1},
        {"context_condition": "disambig", "stereo_idx": 1, "counter_idx": 2, "unknown_idx": 0, "correct_idx": 2},
        {"context_condition": "disambig", "stereo_idx": 1, "counter_idx": 2, "unknown_idx": 0, "correct_idx": 2},
    ]
    ds = Dataset.from_list(recs)
    preds = [1, 1, 1, 1]  # correct on stereo, wrong on counter => acc = 0.5, bias = 0.5
    assert bias_adjusted_accuracy_disambig_metric(preds, [], ds, None, None) == pytest.approx(0.0)
