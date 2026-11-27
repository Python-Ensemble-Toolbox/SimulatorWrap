"""Integration: point OPM_ADJOINT_CASE at doc/tutorials/MODEL_1D_DEBUG of OPM/opm-adjoint."""

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from subsurface.multphaseflow.opm_adjoint import FlowAdjoint, _connection_factors_from_permeability
from misc.structures.layout import DataLayout


# Fixed steps: one substep per report. Adaptive steps: several, of which the last one carries the datum.
@pytest.mark.skipif(not os.environ.get("OPM_ADJOINT_CASE"), reason="requires flow_adjoint and its tutorial deck")
@pytest.mark.parametrize("steps,flags", [((2, 4), "--enable-adaptive-time-stepping=false"), ((40, 80), "")])
def test_weighted_producer_rate_matches_forward_directional_difference(tmp_path, monkeypatch, steps, flags):
    source = Path(os.environ["OPM_ADJOINT_CASE"]).resolve()
    deck = (source / "MODEL_1D_DEBUG.DATA").read_text()
    deck = re.sub(r"INCLUDE\s+(\w+\.txt)\s*/", lambda m: (source / m[1]).read_text(), deck)
    deck = re.sub(r"(?ms)^PERMX\s.*?/", "PERMX\n${' '.join(map(str, permx))} /", deck)
    deck = re.sub(r"(?ms)^PORO\s.*?/", "PORO\n${' '.join(map(str, poro))} /", deck)
    deck = re.sub(r"(?ms)^TSTEP\s.*?/", f"TSTEP\n {steps[0]} {steps[1] - steps[0]} /", deck)
    monkeypatch.chdir(tmp_path)
    Path("TEST.mako").write_text(deck)
    sim = FlowAdjoint({"runfile": "TEST", "datatype": ["wopr:prod"], "reporttype": "days", "reportpoint": list(steps),
                       "adjoint_executable": os.environ.get("FLOW_ADJOINT", "flow_adjoint"),
                       "simoptions": {"sim_flag": f"{flags} --tolerance-cnv=1e-7 --tolerance-mb=1e-9"}})
    sim.setup_fwd_run()
    x, d = np.full(3, 100.), np.array([.2, 1., -.3])
    w = np.array([[1., 0., -.5], [0., 1., 1.]])  # two Jacobian rows and a projected row
    layout = DataLayout.from_frame(pd.DataFrame({"wopr:prod": [0., 0.]}, index=list(steps)))
    p = np.full(3, .1)
    g = sim.run_vjp({"permx": x, "poro": p}, layout.to_frame(w))

    def forward(y, z):
        records = sim.run_fwd_sim({"permx": y, "poro": z}, 0)
        return np.array([r["wopr:prod"] for r in records]) @ w

    delta = .2
    fd = (forward(x + delta*d, p) - forward(x - delta*d, p)) / (2*delta)
    assert np.linalg.norm(fd) > 1e-5
    np.testing.assert_allclose(g["permx"] @ d, fd, rtol=.02, atol=1e-5)
    delta = .0002
    fd = (forward(x, p + delta*d) - forward(x, p - delta*d)) / (2*delta)
    assert np.linalg.norm(fd) > 1e-5
    np.testing.assert_allclose(g["poro"] @ d, fd, rtol=.02, atol=1e-5)


def test_the_solver_of_the_adjoint_is_the_users_choice_and_the_run_mode_is_not():
    options = {"runfile": "TEST", "datatype": ["wopr:prod"], "reporttype": "days", "reportpoint": [2, 4]}
    sim = FlowAdjoint({**options, "simoptions": {"sim_flag": "--adjoint-linear-solver=umfpack"}})
    assert "--adjoint-linear-solver=umfpack" in sim.flags
    with pytest.raises(ValueError, match="FlowAdjoint owns"):
        FlowAdjoint({**options, "simoptions": {"sim_flag": "--adjoint-mode=gradient"}})


PERMX = "PERMX\n${' '.join(map(str, permx))} /"


def tutorial_template(steps=(2, 4), flags="--enable-adaptive-time-stepping=false", report_steps_only=False, **blocks):
    """The tutorial deck as a Mako template, with the given keyword blocks replaced."""
    source = Path(os.environ["OPM_ADJOINT_CASE"]).resolve()
    deck = (source / "MODEL_1D_DEBUG.DATA").read_text()
    deck = re.sub(r"INCLUDE\s+(\w+\.txt)\s*/", lambda m: (source / m[1]).read_text(), deck)
    deck = re.sub(r"(?ms)^TSTEP\s.*?/", f"TSTEP\n {steps[0]} {steps[1] - steps[0]} /", deck)
    if report_steps_only:
        deck = deck.replace("SUMMARY\n", "SUMMARY\nRPTONLY\n", 1)
    for keyword, block in blocks.items():
        deck = re.sub(rf"(?ms)^{keyword}\s.*?/", lambda _: block, deck)
    Path("TEST.mako").write_text(deck)
    sim = FlowAdjoint({"runfile": "TEST", "datatype": ["wopr:prod"], "reporttype": "days", "reportpoint": list(steps),
                       "adjoint_executable": os.environ.get("FLOW_ADJOINT", "flow_adjoint"),
                       "simoptions": {"sim_flag": f"{flags} --tolerance-cnv=1e-7 --tolerance-mb=1e-9"}})
    sim.setup_fwd_run()
    layout = DataLayout.from_frame(pd.DataFrame({"wopr:prod": [0., 0.]}, index=list(steps)))
    return sim, layout.to_frame(np.array([[1.], [.5]]))


@pytest.mark.skipif(not os.environ.get("OPM_ADJOINT_CASE"), reason="requires flow_adjoint and its tutorial deck")
def test_an_axis_the_deck_copies_is_found_and_differentiated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The flow is along x, which the state reaches only through the deck's COPY
    sim, weights = tutorial_template(PERMY="PERMY\n${' '.join(map(str, permy))} /\nCOPY\n PERMY PERMX /\n/\nMULTIPLY\n PERMX 0.5 /\n/",
                                     PERMX="")
    report = sim.check_chain_rule({"permy": np.full(3, 200.)}, weights)
    assert report["ties"] == {"permy": {"x": 0.5}} and report["approximate"] == []
    rows = report["finite_difference"]
    assert len(rows) == 2 and all(row["ok"] for row in rows) and all(abs(row["adjoint"]) > 1e-6 for row in rows)


@pytest.mark.skipif(not os.environ.get("OPM_ADJOINT_CASE"), reason="requires flow_adjoint and its tutorial deck")
def test_a_field_the_template_computes_from_the_state_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim, weights = tutorial_template(PERMX=PERMX,
                                     PORO="PORO\n${' '.join(str(0.1 + 1e-4 * k) for k in permx)} /")
    with pytest.raises(ValueError, match="template writes PORO from permx"):
        sim.run_vjp({"permx": np.full(3, 100.)}, weights)


def test_defaulted_connection_factors_are_found_through_includes(tmp_path):
    (tmp_path / "wells.inc").write_text("COMPDAT\n 'P1' 1 1 1 1 OPEN 2* 0.25 /\n 'P2' 2 1 1 1 OPEN 1* 5.3 0.25 /\n"
                                        " 'P3' 3 1 1 1 OPEN 2* 0.25 120.0 /\n/\n")
    (tmp_path / "CASE.DATA").write_text("SCHEDULE\nINCLUDE\n 'wells.inc' /\n")
    assert _connection_factors_from_permeability(tmp_path / "CASE.DATA") == ["P1"]


@pytest.mark.skipif(not os.environ.get("OPM_ADJOINT_CASE"), reason="requires flow_adjoint and its tutorial deck")
def test_a_deck_that_reports_the_summary_at_report_steps_only_gives_the_same_gradient(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gradients = []
    for report_steps_only in (False, True):  # adaptive steps: several substeps per report
        sim, weights = tutorial_template((40, 80), "", report_steps_only, PERMX=PERMX)
        gradients.append(sim.run_vjp({"permx": np.full(3, 100.)}, weights)["permx"])
    assert np.abs(gradients[0]).max() > 0
    np.testing.assert_allclose(gradients[1], gradients[0], rtol=1e-9)
