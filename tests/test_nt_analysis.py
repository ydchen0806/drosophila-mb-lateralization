import pandas as pd

from bio_fly.nt_analysis import compute_fraction_statistics, summarize_nt_by_subtype


def test_summarize_nt_by_subtype() -> None:
    neuron_inputs = pd.DataFrame(
        [
            {"root_id": 1, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "left", "total_input_synapses": 10, "ser_input": 1, "gaba_input": 1, "ach_input": 1, "glut_input": 3, "oct_input": 0, "da_input": 0},
            {"root_id": 2, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "right", "total_input_synapses": 10, "ser_input": 3, "gaba_input": 1, "ach_input": 1, "glut_input": 1, "oct_input": 0, "da_input": 0},
        ]
    )
    summary, effects = summarize_nt_by_subtype(neuron_inputs)

    assert len(summary) == 2
    ser = effects[effects["nt"] == "ser"].iloc[0]
    assert ser["right_laterality_index"] == 0.5


def test_compute_fraction_statistics() -> None:
    neuron_inputs = pd.DataFrame(
        [
            {"root_id": 1, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "left", "ser_fraction": 0.1, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.3, "oct_fraction": 0.0, "da_fraction": 0.0},
            {"root_id": 2, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "left", "ser_fraction": 0.2, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.3, "oct_fraction": 0.0, "da_fraction": 0.0},
            {"root_id": 3, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "right", "ser_fraction": 0.4, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.1, "oct_fraction": 0.0, "da_fraction": 0.0},
            {"root_id": 4, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "right", "ser_fraction": 0.5, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.1, "oct_fraction": 0.0, "da_fraction": 0.0},
        ]
    )
    stats = compute_fraction_statistics(neuron_inputs, n_boot=10)

    ser = stats[stats["nt"] == "ser"].iloc[0]
    assert ser["right_minus_left_fraction"] > 0
    assert "fdr_q" in stats.columns


def test_compute_fraction_statistics_handles_constant_groups() -> None:
    neuron_inputs = pd.DataFrame(
        [
            {"root_id": 1, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "left", "ser_fraction": 0.1, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.3, "oct_fraction": 0.0, "da_fraction": 0.0},
            {"root_id": 2, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "left", "ser_fraction": 0.1, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.3, "oct_fraction": 0.0, "da_fraction": 0.0},
            {"root_id": 3, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "right", "ser_fraction": 0.1, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.6, "oct_fraction": 0.0, "da_fraction": 0.0},
            {"root_id": 4, "hemibrain_type": "KCapbp", "cell_type": "KC", "side": "right", "ser_fraction": 0.1, "gaba_fraction": 0.1, "ach_fraction": 0.1, "glut_fraction": 0.6, "oct_fraction": 0.0, "da_fraction": 0.0},
        ]
    )
    stats = compute_fraction_statistics(neuron_inputs, n_boot=10)

    ser = stats[stats["nt"] == "ser"].iloc[0]
    glut = stats[stats["nt"] == "glut"].iloc[0]
    assert ser["welch_p"] == 1.0
    assert glut["welch_p"] == 0.0
