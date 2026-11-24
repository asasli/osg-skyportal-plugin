"""
ASTRANet bridge — turns a SkyPortal spectrum-analysis payload into an ASTRANet run.

Takes a SkyPortal payload (``spectra`` + ``redshift`` CSV plus free-form
``analysis_parameters``), parses every spectrum of the source and hands them to
``astranet.skyportal.analyze``, which lives in the ASTRANet package next to the
models it uses. The result carries the class probabilities of the twelve-member
ensemble, the object-level novelty stage (p-value and alarms at 1, 5, 10 and
15 %), flat annotations and a classification on the Sitewide Taxonomy.

Stdlib only at import time: the payload is all CSV, so tests import this in a
lightweight env (same reason as snid_bridge/ngsf_bridge).
"""

from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path

# A spectrum's wavelengths/fluxes arrive as one big list-repr per CSV cell.
csv.field_size_limit(10**9)

IMAGE_MODELS = "/opt/astranet/models"

DEFAULTS = {
    # Use every spectrum of the source; a number selects one row, as in snid_bridge.
    "spectrum_index": None,
    "use_redshift": True,
    # One-page PNG report (spectra, class probabilities, member votes, novelty) for the analysis page.
    "plot": True,
}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _read_csv(value) -> list[dict]:
    """SkyPortal ships each input type as a CSV string."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value if "\n" in value else Path(value).read_text()
        return list(csv.DictReader(io.StringIO(text)))
    if isinstance(value, list):
        return value
    return []


def _as_floats(cell) -> list[float]:
    """A wavelength/flux cell arrives as a list repr; float() keeps bare nan/inf."""
    if isinstance(cell, (list, tuple)):
        return [float(x) for x in cell]
    if isinstance(cell, str):
        return [float(x) for x in cell.strip().strip("[]").split(",") if x.strip()]
    raise ValueError(f"cannot read spectrum column of type {type(cell).__name__}")


def _to_float(value) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def parse_spectra(payload: dict) -> list[dict]:
    """Every spectrum of the payload, or the one row ``spectrum_index`` names."""
    rows = _read_csv(payload.get("spectra"))
    if not rows:
        raise ValueError(
            "no spectra in payload; the analysis service needs input_data_type 'spectra'"
        )
    index = _params(payload).get("spectrum_index")
    if index not in (None, ""):
        i = int(index)
        if not -len(rows) <= i < len(rows):
            raise ValueError(f"spectrum_index {i} out of range ({len(rows)} spectra)")
        rows = [rows[i]]
    spectra = []
    for i, row in enumerate(rows):
        lam, flux = _as_floats(row["wavelengths"]), _as_floats(row["fluxes"])
        if len(lam) != len(flux):
            raise ValueError(f"spectrum {i} has {len(lam)} wavelengths but {len(flux)} fluxes")
        spectra.append(
            {
                "id": row.get("id") or i,
                "observed_at": row.get("observed_at"),
                "instrument": row.get("instrument_name") or row.get("origin"),
                "wavelengths": lam,
                "fluxes": flux,
            }
        )
    return spectra


def resolve_redshift(payload: dict) -> float | None:
    """analysis_parameters.redshift wins; otherwise the source's SkyPortal value."""
    override = _to_float(_params(payload).get("redshift"))
    if override is not None:
        return override
    rows = _read_csv(payload.get("redshift"))
    return _to_float(rows[0].get("redshift")) if rows else None


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    from astranet.skyportal import analyze  # heavy import (torch); only in the image

    params = _params(payload)
    # The image keeps its models here; set explicitly so the job works when the image environment is not applied.
    if not params.get("models_dir") and not os.environ.get("ASTRANET_MODELS") and Path(IMAGE_MODELS).is_dir():
        params["models_dir"] = IMAGE_MODELS
    return analyze(parse_spectra(payload), resolve_redshift(payload), params, resource_id, work_dir=work_dir)
