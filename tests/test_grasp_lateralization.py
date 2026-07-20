import numpy as np
import pandas as pd

from bio_fly.grasp_lateralization import GraspAnalysisConfig, analyze_fly_level, prepare_fly_level


def test_prepare_fly_level_averages_technical_repeats() -> None:
    raw = pd.DataFrame(
        [
            {"fly_id": "f1", "group": "experimental", "batch": "b1", "left_signal": 10, "right_signal": 20},
            {"fly_id": "f1", "group": "experimental", "batch": "b1", "left_signal": 14, "right_signal": 22},
        ]
    )
    fly = prepare_fly_level(raw)
    assert len(fly) == 1
    assert fly.loc[0, "n_technical_repeats"] == 2
    assert fly.loc[0, "left_signal"] == 12
    assert fly.loc[0, "right_signal"] == 21


def test_direction_variable_lateralization_requires_control_separation() -> None:
    experimental_li = np.array([-0.40, -0.32, -0.25, 0.24, 0.31, 0.42])
    control_li = np.array([-0.03, -0.02, 0.00, 0.01, 0.02, 0.04])
    rows = []
    for group, values in [("experimental", experimental_li), ("control", control_li)]:
        for index, value in enumerate(values):
            rows.append(
                {
                    "fly_id": f"{group}_{index}",
                    "group": group,
                    "left_signal": 1 - value,
                    "right_signal": 1 + value,
                }
            )
    fly = prepare_fly_level(pd.DataFrame(rows))
    summary = analyze_fly_level(
        fly,
        GraspAnalysisConfig(bootstrap_repeats=2_000, permutation_repeats=2_000, random_seed=7),
    )
    assert summary["interpretation"] == "direction_variable_lateralization_above_control"
    assert summary["right_lateralized_flies"] == 3
    assert summary["left_lateralized_flies"] == 3


def test_population_right_shift_can_include_left_lateralized_flies() -> None:
    values = np.array([-0.05, 0.18, 0.22, 0.25, 0.31, 0.35, 0.40, 0.44])
    raw = pd.DataFrame(
        {
            "fly_id": [f"f{i}" for i in range(len(values))],
            "left_signal": 1 - values,
            "right_signal": 1 + values,
        }
    )
    fly = prepare_fly_level(raw)
    summary = analyze_fly_level(
        fly,
        GraspAnalysisConfig(control_group=None, bootstrap_repeats=2_000, permutation_repeats=2_000, random_seed=9),
    )
    assert summary["interpretation"] == "population_right_shifted_with_individual_variability"
    assert summary["left_lateralized_flies"] == 1
