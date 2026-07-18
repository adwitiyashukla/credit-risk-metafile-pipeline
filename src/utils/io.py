"""Shared IO and logging helpers for the pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_json(path: Path) -> dict:
    """Load a JSON file, raising a clear error on missing/corrupt input.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not valid JSON or not a JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")

    return data


def parse_dpd(value: object) -> int:
    """Convert a bureau DPD field to an int.

    Bureaus report DPD as zero-padded strings ("015") but also use
    non-numeric codes ("XXX", "STD") for not-reported periods; those are
    treated as 0.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
