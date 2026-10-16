"""Pure-logic tests for jax_snana_bridge (no jax-snana needed; the fitter is
imported lazily inside run_from_skyportal_inputs). The end-to-end SALT fit is
validated against the real package separately."""

import jax_snana_bridge as jb


def _csv(rows, header="mjd,filter,flux,fluxerr,zp,magsys"):
    return "\n".join([header, *rows])


def test_params_merge_defaults_with_overrides():
    p = jb._params({"analysis_parameters": {"source": "BAYESN.G26", "mwebv": 0.1}})
    assert p["source"] == "BAYESN.G26" and p["mwebv"] == 0.1
    assert p["n_fit_iterations"] == jb.DEFAULTS["n_fit_iterations"]  # default kept


def test_band_map_covers_ztf_and_atlas():
    assert jb.BAND_MAP["ztfg"] == "p48g" and jb.BAND_MAP["ztfr"] == "p48r"
    assert jb.BAND_MAP["atlasc"] == "c_ATLAS" and jb.BAND_MAP["atlaso"] == "o_ATLAS"


def test_lightcurve_arrays_keeps_mapped_bands_with_flux():
    csv = _csv(
        [
            "60000.0,ztfg,100.0,5.0,23.9,ab",  # kept
            "60001.0,ztfr,150.0,6.0,23.9,ab",  # kept
            "60002.0,sdssg,200.0,7.0,23.9,ab",  # unmapped filter -> dropped
            "60003.0,ztfi,,,,ab",  # no flux -> dropped
            "60004.0,ztfr,80.0,0,23.9,ab",  # zero error -> dropped
        ]
    )
    mjd, flux, ferr, band = jb.lightcurve_arrays({"photometry": csv})
    assert band == ["ztfg", "ztfr"]
    assert flux == [100.0, 150.0] and ferr == [5.0, 6.0]


def test_resolve_redshift_param_wins_then_csv():
    assert jb._resolve_redshift({"analysis_parameters": {"redshift": 0.1}}) == 0.1
    assert jb._resolve_redshift({"redshift": "redshift\n0.2\n"}) == 0.2
    assert jb._resolve_redshift({}) is None
