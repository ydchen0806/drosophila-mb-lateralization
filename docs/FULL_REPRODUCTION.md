# Full-pipeline reproduction

The default `make reproduce` target is self-contained and rebuilds all display
items from released source-data tables. The commands below rerun earlier stages
when the large upstream resources are available.

## Expected external-data layout

```text
data/
  raw/zenodo_10676866/proofread_connections_783.feather
  raw/zenodo_10676866/flywire_synapses_783.feather
  raw/zenodo_10877326/sk_lod1_783_healed_ds2.parquet
  processed/flywire_neuron_annotations.parquet
  external/flywire_annotations_upstream/supplemental_files/
    Supplemental_file1_neuron_annotations.tsv
  external/shiu_drosophila_brain_model/Connectivity_783.parquet
  external/male_cns_2025/supplemental_data/mcns_fw_edge_comp.feather
```

Set `BIO_FLY_PROJECT_ROOT` to the repository root when using a nonstandard
layout. Every principal script also accepts explicit input and output paths.

## KC neurotransmitter-prediction analysis

```bash
python scripts/analyze_kc_nt_lateralization.py \
  --connections data/raw/zenodo_10676866/proofread_connections_783.feather \
  --output-dir outputs/kc_nt_lateralization
```

## DPM functional-imaging reanalysis

After obtaining permissioned source workbooks:

```bash
python scripts/analyze_dpm_functional_imaging_lateralization.py \
  --old-xlsx data/functional_imaging/dpm_lateralization/2026-05-21_dpm_lateralization_functional_imaging.xlsx \
  --new-xlsx data/functional_imaging/dpm_lateralization/2026-06-03_dpm_lateralization_functional_imaging.xlsx \
  --output-dir outputs/dpm_functional_imaging
```

## Paired GRASP structural analysis

The input must contain brain-registered left/right measurements. Negative
controls are required to distinguish absolute biological lateralization from
measurement noise.

```bash
python scripts/analyze_grasp_lateralization.py \
  --input /absolute/path/to/grasp_measurements.csv \
  --output-dir outputs/grasp_lateralization
```

The analysis reports signed population direction, right/left individual counts,
absolute laterality and control-based permutation tests separately.

## Whole-brain DNa02 steering model

Install the Torch extra, then run on CUDA or change `--device` to `cpu`:

```bash
python -m pip install -e '.[wholebrain]'
python scripts/run_lateralized_steering.py \
  --annotation-path data/external/flywire_annotations_upstream/supplemental_files/Supplemental_file1_neuron_annotations.tsv \
  --connectivity-path data/external/shiu_drosophila_brain_model/Connectivity_783.parquet \
  --kc-nt-inputs-path outputs/kc_nt_lateralization/kc_neuron_nt_inputs.parquet \
  --output-dir outputs/lateralized_steering \
  --device cuda:0
```

## Associative memory-to-steering replay

```bash
python scripts/run_associative_steering.py \
  --annotation-path data/external/flywire_annotations_upstream/supplemental_files/Supplemental_file1_neuron_annotations.tsv \
  --connectivity-path data/external/shiu_drosophila_brain_model/Connectivity_783.parquet \
  --kc-nt-inputs-path outputs/kc_nt_lateralization/kc_neuron_nt_inputs.parquet \
  --male-edge-component-path data/external/male_cns_2025/supplemental_data/mcns_fw_edge_comp.feather \
  --output-dir outputs/associative_steering \
  --device cuda:0
```

## Arbor cable-cell validation

Install Arbor 0.11.0 and provide the upstream morphology, synapse and screen
tables described by `--help`:

```bash
python -m pip install -e '.[arbor]'
python scripts/run_arbor_slide16_17_glomerulus_combo_compare.py --help
```

The published cable-cell source-data tables are included so figure reproduction
does not require rerunning this high-cost stage.

## Rebuild compact tables and figures

After full simulations finish:

```bash
python scripts/build_model_causal_triangulation.py
python scripts/build_associative_memory_steering.py
make reproduce
```

The model output is a signed relative command. It is not calibrated to firing
rate, angular velocity, T-maze choice or free behavior.
