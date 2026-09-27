from pathlib import Path

import numpy as np
import pytest

from geodesy_toolkit import levelling, rinex_qc
from geodesy_toolkit.levelling import Observation

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_rinex3_gap_and_completeness():
    rep = rinex_qc.check_file(EXAMPLES / "DEMO00IND_R_20262440000_01H_30S_MO.rnx")
    assert rep.marker == "DEMO"
    assert rep.interval_s == 30
    assert rep.epochs == 110
    assert rep.expected_epochs == 120
    assert rep.gap_count == 1


def test_rinex2_complete_file():
    rep = rinex_qc.check_file(EXAMPLES / "dem22440.26o")
    assert rep.version == "2.11"
    assert rep.epochs == 120
    assert rep.completeness_pct == 100.0
    assert rep.gap_count == 0


def test_folder_scan_finds_both_files():
    assert len(rinex_qc.check_path(EXAMPLES)) == 2


def test_error_free_loop_adjusts_exactly():
    fixed = {"BM1": 100.0}
    obs = [Observation("BM1", "A", 1.0, 1.0), Observation("A", "B", 2.0, 1.0), Observation("B", "BM1", -3.0, 1.0)]
    res = levelling.adjust(obs, fixed)
    assert np.allclose(res.heights_m, [101.0, 103.0])
    assert np.allclose(res.residuals_mm, 0, atol=1e-9)


def test_misclosure_is_spread_by_weight():
    fixed = {"BM1": 100.0}
    # 6 mm misclosure around a loop of three equal legs: each residual should be -2 mm
    obs = [Observation("BM1", "A", 1.002, 1.0), Observation("A", "B", 2.002, 1.0), Observation("B", "BM1", -2.998, 1.0)]
    res = levelling.adjust(obs, fixed)
    assert np.allclose(res.residuals_mm, [-2, -2, -2], atol=1e-9)


def test_example_blunder_is_flagged():
    obs = levelling.read_observations(EXAMPLES / "levelling_observations.csv")
    res = levelling.adjust(obs, levelling.read_fixed(EXAMPLES / "levelling_fixed.csv"), sigma_mm_per_km=1.0)
    flagged = [(obs[k].frm, obs[k].to) for k in res.flagged]
    assert flagged == [("B", "C")]
    assert not res.global_test_passed


def test_underdetermined_network_raises():
    # two unknown heights, one observation
    with pytest.raises(ValueError):
        levelling.adjust([Observation("A", "B", 1.0, 1.0)], {"BM1": 0.0})


def test_helmert_height_close_to_normal_height():
    # C = 980 gpu with g = 979 gal gives about 1001 m; the Helmert term changes it by only a few cm
    h = levelling.helmert_height(980.0, 979.0)
    assert h == pytest.approx(980.0 / 0.979, abs=0.1)
    assert h < 980.0 / 0.979
