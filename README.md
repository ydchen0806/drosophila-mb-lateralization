# Drosophila mushroom-body lateralization

[![reproduce](https://github.com/ydchen0806/drosophila-mb-lateralization/actions/workflows/reproduce.yml/badge.svg)](https://github.com/ydchen0806/drosophila-mb-lateralization/actions/workflows/reproduce.yml)

Code and figure source data for the manuscript **Chemical lateralization of the
*Drosophila* mushroom body biases learned steering in a connectome-constrained
model**.

The repository is a focused release extracted from a larger simulation
workspace. It contains the analysis and model code used for the manuscript,
compact source-data tables for every displayed panel, tests for the central
counterfactual operations, and deterministic scripts that regenerate all main
figures.

## Evidence boundary

The repository separates four evidence levels:

1. transmitter-predicted structure in one female FlyWire connectome;
2. fly-level DPM-pathway-evoked KC 5-HT-sensor imaging;
3. deterministic interventions in point, cable-cell and signed-graph models;
4. route context from male CNS, BANC and MANC.

Model seeds and odor panels measure computational robustness, not biological
replication. The modeled DNa02 readout is a relative steering-command
prediction, not measured animal turning. The 6.98-fold left/right associative
command ratio is the registered structural baseline under a symmetrized
chemical gate; the measured chemical gate adds a smaller retrieval-stage
modulation. A rightward sign in one reference connectome is not assumed to be a
species-wide invariant: paired GRASP measurements must be analyzed for both
signed direction and absolute lateralization at the fly level.

## Quick reproduction

Python 3.10 is the reference environment.

```bash
git clone https://github.com/ydchen0806/drosophila-mb-lateralization.git
cd drosophila-mb-lateralization
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
make reproduce
```

`make reproduce` performs three operations:

- verifies SHA-256 hashes and recomputes headline values from `data/source_data`;
- regenerates three main figures and the external-connectome Extended Data figure;
- runs the focused unit tests.

Generated files are written to `figures/` as vector PDF and 300-dpi PNG.

## Repository map

```text
data/source_data/       compact tables underlying displayed panels
data/source_data.sha256 immutable hashes for the released tables
figures/                deterministic figure outputs
scripts/                figure, analysis and full-model entry points
src/bio_fly/            KC, neurotransmitter, Arbor and steering model code
tests/                  focused tests for statistical and causal operations
docs/                   data provenance and full-pipeline instructions
```

## Common commands

```bash
make validate     # audit hashes and headline values
make figures      # regenerate all paper figures
make test         # run focused unit tests
make reproduce    # run all three targets
```

For a fully pinned lightweight environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For the whole-brain Torch backend or Arbor cable-cell backend:

```bash
python -m pip install -e '.[wholebrain]'
python -m pip install -e '.[arbor]'
```

Full raw-data commands, public upstream resources and expected local paths are
documented in [docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md) and
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). The extraction point and
upstream code revisions are recorded in
[docs/CODE_PROVENANCE.md](docs/CODE_PROVENANCE.md).

Direction-agnostic GRASP statistics and input requirements are documented in
[docs/GRASP_LATERALIZATION_ANALYSIS_CN.md](docs/GRASP_LATERALIZATION_ANALYSIS_CN.md).

## Functional-imaging data

Fly-level derived values used in the figures are included. The two source Excel
workbooks are not redistributed because they were provided by collaborators and
require author approval for public raw-data release. Once those files are placed
at the documented paths, `scripts/analyze_dpm_functional_imaging_lateralization.py`
rebuilds the functional-imaging summaries.

## License and citation

Code is released under the MIT License. The compact source-data tables retain
their scientific provenance and should be cited with the associated manuscript
and upstream datasets. Citation metadata are provided in `CITATION.cff`.
