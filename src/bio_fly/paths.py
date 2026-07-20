import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("BIO_FLY_PROJECT_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
REPO_ROOT = PROJECT_ROOT


def _first_existing_path(*candidates: Path, marker: str = "") -> Path:
    for candidate in candidates:
        check_path = candidate / marker if marker else candidate
        if check_path.exists():
            return candidate
    return candidates[0]


DATA_ROOT = _first_existing_path(
    PROJECT_ROOT / "data",
    REPO_ROOT / "data",
    marker="processed/flywire_neuron_annotations.parquet",
)
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
EXTERNAL_ROOT = _first_existing_path(
    PROJECT_ROOT / "data" / "external" / "shiu_drosophila_brain_model",
    PROJECT_ROOT / "external" / "Drosophila_brain_model",
    marker="Connectivity_783.parquet",
)
FLYWIRE_ANNOTATION_ROOT = _first_existing_path(
    PROJECT_ROOT / "data" / "external" / "flywire_annotations",
    PROJECT_ROOT / "external" / "flywire_annotations",
)
DEFAULT_COMPLETENESS_PATH = EXTERNAL_ROOT / "Completeness_783.csv"
DEFAULT_CONNECTIVITY_PATH = EXTERNAL_ROOT / "Connectivity_783.parquet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
