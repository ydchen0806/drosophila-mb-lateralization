from __future__ import annotations

import importlib
import os


class ArborUnavailableError(RuntimeError):
    """Raised when an Arbor-backed run is requested but Arbor is unavailable."""


def import_arbor():
    try:
        return importlib.import_module("arbor")
    except ModuleNotFoundError as exc:
        raise ArborUnavailableError(
            "The Arbor Python package is not installed in this environment. "
            "Install Arbor before running BioFly Arbor experiments. The BioFly "
            "Arbor route is CPU-only; use an Arbor build with multicore support."
        ) from exc


def resolve_thread_count(value: str | int | None, *, cpu_count: int | None = None) -> int:
    available = int(cpu_count or os.cpu_count() or 1)
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return max(1, available - 1)
    threads = int(value)
    if threads < 1:
        raise ValueError("--threads must be 'auto' or a positive integer.")
    return min(threads, available)


def arbor_version(arbor_module) -> str:
    config = getattr(arbor_module, "config", None)
    if callable(config):
        try:
            values = config()
        except Exception:
            values = {}
        version = values.get("version") or values.get("ARB_VERSION")
        if version:
            return str(version)
    return str(getattr(arbor_module, "__version__", "unknown"))
