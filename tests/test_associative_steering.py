import numpy as np

from bio_fly.associative_steering import (
    build_side_restricted_code,
    learned_mbon32_depression,
    signed_dna02_learning_delta,
)


def test_learned_depression_is_training_odor_specific() -> None:
    training = np.array([0.6, 0.4, 0.0, 0.0])
    same = np.array([0.5, 0.5, 0.0, 0.0])
    novel = np.array([0.0, 0.0, 0.5, 0.5])
    weights = np.array([2.0, 1.0, 3.0, 4.0])
    assert learned_mbon32_depression(training, same, weights, 0.75) > 0
    assert learned_mbon32_depression(training, novel, weights, 0.75) == 0


def test_mbon32_depression_drives_contraversive_dna02_delta() -> None:
    assert signed_dna02_learning_delta("left", 2.0, -40.0) == 80.0
    assert signed_dna02_learning_delta("right", 2.0, -10.0) == -20.0


def test_side_restricted_code_has_fixed_side_support_and_unit_mass() -> None:
    activity = np.array([[1.0, 0.5, 0.8, 0.3]])
    gate = np.array([0.2, -0.2, 0.1, -0.1])
    side = np.array(["left", "left", "right", "right"])
    code = build_side_restricted_code(
        activity,
        gate,
        side,
        "left",
        gate_amplitude=0.25,
        active_fraction=0.5,
    )
    assert np.isclose(code.sum(), 1.0)
    assert np.all(code[:, side == "right"] == 0)
