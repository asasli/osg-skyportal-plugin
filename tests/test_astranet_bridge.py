"""Pure-logic tests for astranet_bridge (no torch, no weights): the SkyPortal spectra
wire format, spectrum selection and redshift resolution."""

import csv
import io

import pytest

import astranet_bridge as ab

SPECTRA_COLUMNS = ["observed_at", "wavelengths", "fluxes", "origin"]


def _csv(rows, columns):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


def _payload(n=2, **params):
    rows = [
        {
            "observed_at": f"2026-11-0{i + 1}T00:00:00",
            "wavelengths": str([4000.0 + j for j in range(5)]),
            "fluxes": str([1.0, 2.0, float("nan"), 4.0, 5.0 + i]),
            "origin": "SEDM",
        }
        for i in range(n)
    ]
    return {
        "spectra": _csv(rows, SPECTRA_COLUMNS),
        "redshift": "redshift\n0.05\n",
        "analysis_parameters": params,
    }


def test_all_spectra_parsed_with_nan_kept():
    spectra = ab.parse_spectra(_payload())
    assert len(spectra) == 2 and spectra[0]["instrument"] == "SEDM"
    assert len(spectra[1]["wavelengths"]) == 5 and spectra[1]["fluxes"][2] != spectra[1]["fluxes"][2]


def test_spectrum_index_selects_one_row():
    spectra = ab.parse_spectra(_payload(spectrum_index=-1))
    assert len(spectra) == 1 and spectra[0]["fluxes"][-1] == 6.0
    with pytest.raises(ValueError):
        ab.parse_spectra(_payload(spectrum_index=5))


def test_redshift_override_and_missing():
    assert ab.resolve_redshift(_payload()) == 0.05
    assert ab.resolve_redshift(_payload(redshift=0.2)) == 0.2
    assert ab.resolve_redshift({"spectra": "", "redshift": "redshift\n\n"}) is None


def test_no_spectra_is_an_error():
    with pytest.raises(ValueError):
        ab.parse_spectra({"spectra": _csv([], SPECTRA_COLUMNS)})


def test_length_mismatch_is_an_error():
    bad = {"spectra": _csv([{"wavelengths": "[1, 2, 3]", "fluxes": "[1, 2]"}], SPECTRA_COLUMNS)}
    with pytest.raises(ValueError):
        ab.parse_spectra(bad)
