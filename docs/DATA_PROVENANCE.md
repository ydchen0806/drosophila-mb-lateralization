# Data provenance

## Released source-data tables

`data/source_data/` contains compact tables used directly by the figure scripts.
Their byte-level hashes are fixed in `data/source_data.sha256`. The tables are
derived data, not additional biological replicates.

| Evidence layer | Biological or computational unit | Released tables |
|---|---|---|
| KC chemical-label discovery | KCs and KC subtypes nested in one FlyWire brain | `subtype_*`, `kc_5ht_*`, `whole_brain_*` |
| DPM functional imaging | fly; 29 flies in two batches | `dpm_29fly_*`, `dpm_batch_*`, `dpm_timecourse_*` |
| point and cable-cell models | fixed seed or odor panels | `model_causal_*` |
| DNa02 and associative replay | fixed seed-odor panels | `lateralized_*`, `associative_*` |
| external route context | connectome-derived route/type | `male_cns_*`, `banc_*`, `manc_*` |

## Public upstream resources

- FlyWire v783 proofread connections and synapses: Zenodo record 10676866,
  <https://zenodo.org/records/10676866>
- FlyWire skeleton release: Zenodo record 10877326,
  <https://zenodo.org/records/10877326>
- FlyWire annotations:
  <https://github.com/flyconnectome/flywire_annotations>
- Shiu *Drosophila* brain model and signed connectivity:
  <https://github.com/philshiu/Drosophila_brain_model>

Male CNS, BANC and MANC analyses use the releases cited in the manuscript. They
provide route conservation or downstream context; they do not independently
replicate the side-resolved KC neurotransmitter-prediction endpoint.

## Files not redistributed

The collaborator-provided DPM functional-imaging Excel workbooks are not
included. The release contains fly-level and time-course source data used in the
figures, plus the exact parser/reanalysis script. Public release of the source
workbooks requires confirmation from the experimental data owners.

Large public connectomes, skeletons and signed connectivity matrices are not
duplicated in Git because they total several gigabytes. Download them from the
upstream archives and retain their upstream licenses and citations.
