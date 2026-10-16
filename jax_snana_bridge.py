"""
jax-snana bridge — SALT/BayeSN light-curve fit of a SkyPortal source on an OSG
worker.

Uses jax-snana's ``LightCurveFitter`` Python API with its built-in filter
calibration, so no KCOR file, SNANA-FITS conversion or namelist is needed: the
fit runs straight from arrays. Returns the fit parameters (SALT: x1, c, mB, t0;
BayeSN: THETA, AV) as a SkyPortal result plus a per-filter model light curve for
the photometry-plot overlay, matching the fiesta/redback bridges' contract.

Kept importable without jax-snana (the fitter is imported lazily) so the listener
and the tests can read this module without the heavy dependency.
"""

from __future__ import annotations

import csv
import io
import json  # noqa: F401 — kept for parity with the other bridges' result files
import math
from pathlib import Path  # noqa: F401
from typing import Any

# SkyPortal filter name -> jax-snana built-in filter name. ZTF/P48 and ATLAS are
# in the bundled catalogue; extend this map to add more built-in bands.
BAND_MAP = {
    "ztfg": "p48g",
    "ztfr": "p48r",
    "ztfi": "p48i",
    "atlasc": "c_ATLAS",
    "atlaso": "o_ATLAS",
}

# SkyPortal flux photometry (format=flux, magsys=ab) is micro-Jy AB on this
# zeropoint: mag = -2.5 log10(flux) + PHOT_ZP.
PHOT_ZP = 23.9

DEFAULTS: dict[str, Any] = {
    "source": "SALT3.DOVEKIE",
    "fitwin_trest": (-15.0, 45.0),  # rest-frame phase window the fit uses
    "n_fit_iterations": 3,
}

# Fit-result columns surfaced to SkyPortal (present depends on the model family).
_PARAM_COLS = (
    "PKMJD",
    "PKMJD_err",
    "x1",
    "x1_err",
    "c",
    "c_err",
    "mB",
    "mB_err",
    "x0",
    "x0_err",
    "THETA",
    "THETA_err",
    "AV",
    "AV_err",
    "chi_squared",
    "ndof",
)


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # f != f is True only for NaN


def _read_csv(src) -> list[dict]:
    if not src:
        return []
    text = src if isinstance(src, str) else str(src)
    return list(csv.DictReader(io.StringIO(text)))


def _resolve_redshift(payload: dict):
    """analysis_parameters.redshift wins; else the source's SkyPortal value."""
    z = _to_float(_params(payload).get("redshift"))
    if z is not None:
        return z
    rows = _read_csv(payload.get("redshift"))
    return _to_float(rows[0].get("redshift")) if rows else None


def lightcurve_arrays(payload: dict):
    """SkyPortal flux photometry -> (mjd, flux, flux_err, band) for mapped filters.
    Only finite flux with a positive error in a built-in band is kept."""
    mjd, flux, ferr, band = [], [], [], []
    for r in _read_csv(payload.get("photometry")):
        f = str(r.get("filter", "")).strip().lower()
        if f not in BAND_MAP:
            continue
        t, fl, fe = _to_float(r.get("mjd")), _to_float(r.get("flux")), _to_float(r.get("fluxerr"))
        if t is None or fl is None or fe is None or fe <= 0:
            continue
        mjd.append(t)
        flux.append(fl)
        ferr.append(fe)
        band.append(f)
    return mjd, flux, ferr, band


def _model_lightcurve(fitter, result, bands: dict, t0: float, redshift: float) -> dict:
    """Per-filter apparent-mag model curve on an MJD grid, via the fitter's
    predict() at the best-fit parameters, for the SkyPortal photometry overlay.
    Shape matches the other bridges: {filter: [[mjd, med, lo, hi], ...]}."""
    import numpy as np

    z = redshift or 0.0
    grid = np.linspace(t0 - 20.0 * (1.0 + z), t0 + 50.0 * (1.0 + z), 60)

    def _to_mag(f):
        return -2.5 * math.log10(f) + PHOT_ZP if (np.isfinite(f) and f > 0) else None

    out: dict[str, list] = {}
    for sp_band, jx_band in bands.items():
        try:
            # predict returns a dict (or 0-d array wrapping one) with 'flux' and
            # 'model_flux_err' on the supplied zeropoint.
            pred = fitter.predict(result, grid, np.array([jx_band] * len(grid)), zp=PHOT_ZP)
            if not isinstance(pred, dict):
                pred = np.asarray(pred).reshape(-1)[0]
            flux = np.asarray(pred["flux"]).reshape(-1)
            ferr = np.asarray(pred.get("model_flux_err", np.zeros_like(flux))).reshape(-1)
        except Exception:  # noqa: BLE001 — overlay is best-effort, per band
            continue
        rows = []
        for i in range(len(grid)):
            med = _to_mag(float(flux[i]))
            if med is None:
                continue
            # flux+err -> brighter (lo mag); flux-err -> fainter (hi mag)
            lo = _to_mag(float(flux[i]) + float(ferr[i])) or med
            hi = _to_mag(max(float(flux[i]) - float(ferr[i]), 1e-30)) or med
            rows.append([float(grid[i]), med, lo, hi])
        if rows:
            out[sp_band] = rows
    return out


def _posterior_samples(res, vals: dict, n_samples: int = 2000) -> dict | None:
    """Draw a Gaussian posterior from the fit's parameter covariance.

    A SALT fit is a least-squares, so its posterior is the multivariate normal
    at the best-fit mean with the fit covariance (jax-snana's param_covariances
    over cov_output_params). x0 is reported as mB via the exact transform the fit
    uses, matching the headline parameters. Shape {param: [samples]} drives
    SkyPortal's corner plot, like the fiesta/redback bridges."""
    import numpy as np

    # param_covariances is square over every fitted parameter (res.params order,
    # e.g. PKMJD/x1/c/x0); cov_output_params is the reportable subset (no PKMJD),
    # so slice that subset's rows/cols out of the full matrix by position.
    full = list(res.params.keys())
    names = [p for p in (getattr(res, "cov_output_params", []) or []) if p in full]
    if not names:
        return None
    try:
        idx = [full.index(p) for p in names]
        mean = np.array([float(np.asarray(res.params[p])[0]) for p in names])
        cov = np.asarray(res.param_covariances[0], dtype=float)[np.ix_(idx, idx)]
    except Exception:  # noqa: BLE001 — posterior is best-effort
        return None
    if cov.shape != (len(names), len(names)) or not (
        np.all(np.isfinite(cov)) and np.all(np.isfinite(mean))
    ):
        return None
    cov = 0.5 * (cov + cov.T)  # symmetrize; a least-squares cov can drift off-symmetric numerically
    rng = np.random.default_rng(0)
    try:
        draws = rng.multivariate_normal(mean, cov, size=n_samples)
    except Exception:  # noqa: BLE001 — fall back to marginals if cov isn't usable
        diag = np.clip(np.diag(cov), 0.0, None)
        draws = rng.normal(mean, np.sqrt(diag), size=(n_samples, len(names)))
    col = {name: draws[:, i] for i, name in enumerate(names)}

    out: dict[str, list] = {}
    for p in ("x1", "c"):
        if p in col:
            out[p] = [float(x) for x in col[p]]
    # x0 -> mB with the offset fixed by the best fit (mB = -2.5 log10 x0 + const).
    x0b, mBb = vals.get("x0"), vals.get("mB")
    if "x0" in col and x0b and x0b > 0 and mBb is not None:
        const = mBb + 2.5 * math.log10(x0b)
        x0d = col["x0"]
        pos = x0d > 0
        if pos.any():
            mB = -2.5 * np.log10(np.where(pos, x0d, 1.0)) + const
            out["mB"] = [float(m) for m, ok in zip(mB, pos) if ok]
            # keep x1/c aligned to the samples that yielded a finite mB
            for p in ("x1", "c"):
                if p in col:
                    out[p] = [float(v) for v, ok in zip(col[p], pos) if ok]
    return out or None


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    """Fit the source's light curve with a jax-snana model. Returns the
    fiesta-style {status, message, source, results, model_lightcurve, n_detections}."""
    import numpy as np
    from jax_snlc_fit import LightCurveFitter

    params = _params(payload)
    source = str(params["source"])
    mjd, flux, ferr, band = lightcurve_arrays(payload)
    if len(mjd) < 3:
        return {
            "status": "failure",
            "message": f"Not enough photometry to fit (have {len(mjd)}).",
            "source": source,
        }
    z = _resolve_redshift(payload)
    if z is None or z <= 0:
        return {
            "status": "failure",
            "message": "A positive redshift is required for the fit.",
            "source": source,
        }
    mwebv = _to_float(params.get("mwebv")) or 0.0

    settings: dict[str, Any] = {}
    if params.get("fitwin_trest"):
        settings["fitwin_trest"] = tuple(params["fitwin_trest"])
    if params.get("n_fit_iterations"):
        settings["n_fit_iterations"] = int(params["n_fit_iterations"])
    fitter = LightCurveFitter(source, batch_size=1, **settings)

    present = {b: BAND_MAP[b] for b in set(band)}  # only the bands this object has
    res = fitter.fit(
        mjd=np.array(mjd),
        flux=np.array(flux),
        flux_err=np.array(ferr),
        zp=PHOT_ZP,
        band=np.array(band),
        redshift=float(z),
        mwebv=mwebv,
        band_map=present,
        progress=False,
    )
    table = res.to_table()
    row = table[0]
    status = str(row["status"])
    vals = {c: _to_float(row[c]) for c in _PARAM_COLS if c in table.colnames}

    if status != "accepted":
        why = str(row["reason_description"]) if "reason_description" in table.colnames else status
        return {
            "status": "failure",
            "message": f"{source} fit {status}: {why}",
            "source": source,
            "results": {"source": source, "status": status, **vals},
        }

    t0 = vals.get("PKMJD") or float(np.median(mjd))
    chi2, ndof = vals.get("chi_squared"), vals.get("ndof")
    red_chi2 = (chi2 / ndof) if (chi2 is not None and ndof) else None
    headline = " ".join(
        f"{k}={vals[k]:.3f}" for k in ("x1", "c", "mB", "THETA", "AV") if vals.get(k) is not None
    )
    msg = f"{source}: {headline}" + (f" (chi2/ndof={red_chi2:.1f})" if red_chi2 is not None else "")

    out = {
        "status": "success",
        "message": msg,
        "source": source,
        "results": {"source": source, "redshift": float(z), "n_detections": len(mjd), **vals},
        "n_detections": len(mjd),
    }
    try:
        out["model_lightcurve"] = _model_lightcurve(fitter, res, present, t0, z)
    except Exception as e:  # noqa: BLE001 — overlay data is best-effort
        out["model_lightcurve_error"] = repr(e)
    try:
        samples = _posterior_samples(res, vals)
        if samples:
            out["posterior_samples"] = samples
    except Exception as e:  # noqa: BLE001 — posterior is best-effort
        out["posterior_samples_error"] = repr(e)
    return out
