import numpy as np
import pandas as pd

from bio_fly.lateralized_steering import (
    extract_dna02_drive,
    paired_condition_contrast,
    subtype_preserving_gate_shuffle,
)


def test_subtype_preserving_shuffle_keeps_each_subtype_multiset() -> None:
    gate = np.array([-1.0, -1.0, 1.0, 1.0, -0.5, 0.5])
    subtype = np.array(["a", "a", "a", "a", "b", "b"])
    shuffled = subtype_preserving_gate_shuffle(gate, subtype, np.random.default_rng(7))
    for label in ["a", "b"]:
        mask = subtype == label
        assert np.array_equal(np.sort(shuffled[mask]), np.sort(gate[mask]))


def test_extract_dna02_drive_aggregates_only_requested_steps() -> None:
    response = pd.DataFrame(
        [
            {"root_id": 10, "score": 0.2, "step": 1},
            {"root_id": 10, "score": 0.3, "step": 2},
            {"root_id": 20, "score": 0.8, "step": 2},
            {"root_id": 20, "score": 2.0, "step": 3},
        ]
    )
    result = extract_dna02_drive(response, {"left": 10, "right": 20}, step_limit=2)
    assert np.isclose(result["dna02_left_drive"], 0.5)
    assert np.isclose(result["dna02_right_drive"], 0.8)
    assert np.isclose(result["dna02_right_minus_left"], 0.3)


def test_paired_condition_contrast_preserves_seed_and_odor_pairing() -> None:
    raw = pd.DataFrame(
        [
            {"seed": 1, "split": "validation", "odor_index": 0, "odor_name": "a", "condition": "real", "dna02_right_minus_left": 0.4},
            {"seed": 1, "split": "validation", "odor_index": 0, "odor_name": "a", "condition": "sym", "dna02_right_minus_left": 0.1},
            {"seed": 2, "split": "validation", "odor_index": 0, "odor_name": "b", "condition": "real", "dna02_right_minus_left": -0.1},
            {"seed": 2, "split": "validation", "odor_index": 0, "odor_name": "b", "condition": "sym", "dna02_right_minus_left": -0.2},
        ]
    )
    paired = paired_condition_contrast(raw, "real", "sym")
    assert np.allclose(paired.sort_values("seed")["delta"], [0.3, 0.1])
