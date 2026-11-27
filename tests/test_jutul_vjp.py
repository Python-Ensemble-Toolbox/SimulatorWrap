from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from test_jutul_darcy import REPORT_DATES, _copy_case_folder, _tiny_folder
from subsurface.multphaseflow.jutul_darcy import JutulDarcy


def test_weighted_rates_match_forward_difference(tmp_path, monkeypatch):
    case = _copy_case_folder(tmp_path, _tiny_folder(), "tiny")
    monkeypatch.chdir(case)
    sim = JutulDarcy({"runfile": "RUNFILE.mako", "reportpoint": REPORT_DATES, "reporttype": "dates",
                      "datatype": ["WOPR:PRO2", "WWPR:PRO1"], "perm_copied": True})
    x = np.log(np.load("PERMX.npy"))
    direction = np.zeros_like(x)
    direction[94] = 1
    assert x[94] + .01 < 6  # differentiating the interior of this clipped template
    labels = [REPORT_DATES[-1], REPORT_DATES[2]]
    weights = pd.DataFrame({"WWPR:PRO1": [None, np.array([.2, -.7])],
                            "WOPR:PRO2": [np.array([1., .5]), None]}, index=labels)
    sim.check_chain_rule({"log_permx": x})  # else the first run_vjp does, with a case of its own per variable
    with patch.object(sim, "_setup_case", wraps=sim._setup_case) as setup:
        g = sim.run_vjp({"log_permx": x}, weights)["log_permx"]
    assert setup.call_count == 1
    single = weights.map(lambda cell: cell[0] if cell is not None else np.nan)
    forecast = JutulDarcy({"runfile": "RUNFILE.mako", "reportpoint": REPORT_DATES, "reporttype": "dates",
                           "datatype": sim.datatype, "perm_copied": True,
                           "weighted_adjoint": {"parameters": ["log_permx"]}})
    forecast.adjoint_weights = lambda output, i: single.loc[output.index.intersection(single.index)]
    _, gradient = forecast.run_fwd_sim({"log_permx": x})
    np.testing.assert_allclose(gradient["log_permx"], g[0], rtol=1e-6, atol=1e-7)

    def f(y):
        frame = sim.run_fwd_sim({"log_permx": y})
        return frame.at[labels[0], "WOPR:PRO2"] * np.array([1., .5]) + frame.at[labels[1], "WWPR:PRO1"] * np.array([.2, -.7])

    fd = (f(x + .01*direction) - f(x - .01*direction)) / .02
    assert np.linalg.norm(fd) > 1e-4
    np.testing.assert_allclose(g @ direction, fd, rtol=.05, atol=1e-4)


def tiny(tmp_path, monkeypatch, **options):
    case = _copy_case_folder(tmp_path, _tiny_folder(), "tiny")
    monkeypatch.chdir(case)
    sim = JutulDarcy({"runfile": "RUNFILE.mako", "reportpoint": REPORT_DATES, "reporttype": "dates",
                      "datatype": ["WOPR:PRO2", "WWPR:PRO1"], **options})
    weights = pd.DataFrame({"WOPR:PRO2": [np.array([1.]), None], "WWPR:PRO1": [None, np.array([-.7])]},
                           index=[REPORT_DATES[-1], REPORT_DATES[2]])
    return sim, np.log(np.load("PERMX.npy")), weights


def test_the_copied_axes_and_the_well_indices_of_the_deck_are_found_and_differentiated(tmp_path, monkeypatch):
    sim, x, weights = tiny(tmp_path, monkeypatch)
    report = sim.check_chain_rule({"log_permx": x}, weights)
    assert report["ties"] == {"log_permx": {"y": 1.0, "z": 1.0}}
    assert report["derived"] == {"log_permx": ["Transmissibilities", "WellIndices"]}
    rows = report["finite_difference"]
    assert len(rows) == 2 and all(row["ok"] for row in rows) and all(abs(row["adjoint"]) > 1e-4 for row in rows)


@pytest.mark.parametrize("options,missing", [
    ({"perm_copied": False}, "PERMY follows log_permx by the factor 1, which the wrapper does not add"),
    ({"well_index_from_perm": False}, r"connection\(s\) of \w+ follows log_permx, which the wrapper does not differentiate"),
])
def test_a_term_the_options_switch_off_is_refused_before_anything_is_simulated(tmp_path, monkeypatch, options, missing):
    sim, x, weights = tiny(tmp_path, monkeypatch, **options)
    with patch.object(sim, "_simulate", wraps=sim._simulate) as simulate, pytest.raises(ValueError, match=missing):
        sim.run_vjp({"log_permx": x}, weights)
    assert simulate.call_count == 0


def test_connection_factors_given_in_the_deck_are_not_differentiated(tmp_path, monkeypatch):
    sim, x, _ = tiny(tmp_path, monkeypatch)
    shutil.copy("include/Schdl_fixed_wi.sch", "include/Schdl.sch")
    assert sim.check_chain_rule({"log_permx": x})["derived"] == {"log_permx": ["Transmissibilities"]}
    sim.well_index_from_perm = True
    with pytest.raises(ValueError, match="is differentiated by the wrapper, but does not follow log_permx"):
        sim.check_chain_rule({"log_permx": x})


def test_a_field_the_template_computes_from_the_state_is_refused(tmp_path, monkeypatch):
    sim, x, _ = tiny(tmp_path, monkeypatch)
    template, copies = re.subn(r"(?m)^ *'PERMX' +'PERMY' */\n", "", Path("RUNFILE.mako").read_text())
    assert copies == 1
    Path("RUNFILE.mako").write_text(template.replace(
        "COPY\n", "PERMY\n${' '.join('%.3f' % np.exp(1.1 * v) for v in log_permx)} /\nCOPY\n", 1))
    with pytest.raises(ValueError, match="the template writes PERMY from log_permx.*PERMY follows log_permx, but not by a constant"):
        sim.check_chain_rule({"log_permx": x})


def test_a_uniform_field_whose_copies_cannot_be_detected_is_settled_by_the_check(tmp_path, monkeypatch):
    sim, x, _ = tiny(tmp_path, monkeypatch)
    uniform = np.full_like(x, x.mean())
    with pytest.warns(UserWarning, match="PERMX is uniform"), pytest.raises(
            ValueError, match="PERMY follows log_permx by the factor 1, which the wrapper does not add"):
        sim.check_chain_rule({"log_permx": uniform})
    sim.perm_copied = True
    assert sim.check_chain_rule({"log_permx": uniform})["ties"] == {"log_permx": {"y": 1.0, "z": 1.0}}
