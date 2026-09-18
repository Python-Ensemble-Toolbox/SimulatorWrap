"""Integration tests for the JutulDarcy wrapper.

These tests execute a real simulation and are intentionally heavier than unit tests.
Run selectively when validating simulator integration.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import os
import shutil
import numpy as np
import pytest

from subsurface.multphaseflow.jutul_darcy import JutulDarcy


REPORT_DATES = [
    datetime(2023, 2, 5),
    datetime(2024, 3, 11),
    datetime(2025, 4, 15),
    datetime(2026, 5, 20),
    datetime(2027, 6, 24),
    datetime(2028, 7, 28),
    datetime(2029, 9, 1),
    datetime(2030, 10, 6),
    datetime(2031, 11, 10),
    datetime(2032, 12, 14),
]

DATA_TYPES = [
    "WOPR:PRO1",
    "WOPR:PRO2",
    "WOPR:PRO3",
    "WWPR:PRO1",
    "WWPR:PRO2",
    "WWPR:PRO3",
    "WWIR:INJ1",
]

GRADIENT_STEP = datetime(2032, 12, 14)
GRADIENT_TARGET = "WOPR:PRO2"


def _tiny_folder() -> Path:
    """Return absolute path to the `Example/TINY` input case directory."""
    tiny_path = Path(__file__).resolve().parents[1] / "Example" / "TINY"
    if not tiny_path.exists():
        raise FileNotFoundError(f"TINY folder not found at: {tiny_path}")
    return tiny_path


@contextmanager
def _working_directory(path: Path):
    """Temporarily change current working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _copy_case_folder(target: Path, source: Path, name: str) -> Path:
    """Copy the TINY case folder into a per-test destination."""
    case_path = target / name
    shutil.copytree(source, case_path)
    return case_path


def _run_case(case_path: Path, options: dict):
    """Run JutulDarcy for the given case path and options."""
    log_permx = np.log(np.load(case_path / "PERMX.npy"))
    simulator = JutulDarcy(options)
    with _working_directory(case_path):
        return simulator({"log_permx": log_permx})


@pytest.fixture(scope="module")
def options() -> dict:
    """Base simulation options shared by tests."""
    return {
        "reporttype": "dates",
        "reportpoint": REPORT_DATES,
        "runfile": "RUNFILE.mako",
        "datatype": DATA_TYPES,
    }


@pytest.fixture(scope="module")
def tiny_folder() -> Path:
    """Path to the immutable source TINY case in repository."""
    return _tiny_folder()


@pytest.fixture(scope="module")
def run_simulation_with_adjoint(tmp_path_factory, options, tiny_folder):
    """Run one adjoint-enabled simulation and share output across gradient tests."""
    base_tmp = tmp_path_factory.mktemp("adjoint")
    case_path = _copy_case_folder(base_tmp, tiny_folder, "TINY_ADJOINT")

    case_options = {
        **options,
        'perm_copied': True, # Include total derivative when PERMX is copied to PERMY and PERMZ
        "adjoints": {
            "WOPR": {
                "steps": [GRADIENT_STEP],
                "wellID": "PRO2",
                "parameters": ["log_permx", "permx"],
            }
        },
    }
    return _run_case(case_path, case_options)


def test_simulation_runs_and_matches_requested_outputs(tmp_path, options, tiny_folder):
    """Simulation returns a non-empty table with all requested columns and dates."""
    case_path = _copy_case_folder(tmp_path, tiny_folder, "TINY")
    results = _run_case(case_path, options)

    assert not results.empty, "Simulation result table is empty"

    missing_columns = sorted(set(options["datatype"]) - set(results.columns))
    assert not missing_columns, f"Missing expected result columns: {missing_columns}"

    missing_dates = [date for date in options["reportpoint"] if date not in results.index]
    assert not missing_dates, f"Missing expected report dates: {missing_dates}"


def test_gradient_contains_expected_structure(run_simulation_with_adjoint, options):
    """Adjoint run returns expected gradient columns and valid report dates."""
    _, gradient = run_simulation_with_adjoint

    assert not gradient.empty, "Gradient table is empty"

    expected_columns = {
        (GRADIENT_TARGET, "log_permx"),
        (GRADIENT_TARGET, "permx"),
    }
    missing_columns = sorted(expected_columns - set(gradient.columns))
    assert not missing_columns, f"Missing expected gradient columns: {missing_columns}"

    unexpected_dates = [date for date in gradient.index if date not in options["reportpoint"]]
    assert not unexpected_dates, f"Gradient has unexpected report dates: {unexpected_dates}"


def test_gradient_log_permx_matches_chain_rule(run_simulation_with_adjoint, tiny_folder):
    """Validate dF/d(log(k)) = dF/d(k) * k for the configured objective and step."""
    _, gradient = run_simulation_with_adjoint

    grad_log_permx = gradient.loc[GRADIENT_STEP, (GRADIENT_TARGET, "log_permx")]
    grad_permx = gradient.loc[GRADIENT_STEP, (GRADIENT_TARGET, "permx")]
    permx = np.load(tiny_folder / "PERMX.npy")

    np.testing.assert_allclose(grad_log_permx, grad_permx * permx)



# --------------------------------------------------------------------------- #
# Well-index chain rule
# --------------------------------------------------------------------------- #
# TINY is a 10 x 10 x 2 box. Wells sit at (i, j) = (1, 1), (5, 1), (10, 1),
# (1, 10), (5, 10) and (10, 10), each perforating both layers, and COMPDAT
# leaves the connection factor defaulted -- so every one of these cells has a
# well index derived from its own permeability.
NX, NY = 10, 10
WELL_IJ = [(1, 1), (5, 1), (10, 1), (1, 10), (5, 10), (10, 10)]
PERFORATED_CELLS = sorted(
    (i - 1) + (j - 1) * NX + k * NX * NY
    for i, j in WELL_IJ
    for k in (0, 1)
)

#: Perforated cells of PRO2 (the objective well) plus one cell far from any well.
FD_CELLS = [94, 194, 55]

#: Relative perturbation used for the central difference.
FD_EPS = 0.01


def _adjoint_options(options: dict, **extra) -> dict:
    """Base options plus an adjoint request for dWOPR:PRO2/dPERMX."""
    return {
        **options,
        "perm_copied": True,
        "adjoints": {
            "WOPR": {
                "steps": [GRADIENT_STEP],
                "wellID": "PRO2",
                "parameters": ["permx"],
            }
        },
        **extra,
    }


@pytest.fixture(scope="module")
def gradient_without_wi_chain_rule(tmp_path_factory, options, tiny_folder):
    """Adjoint gradient with the well-index chain rule switched off."""
    base_tmp = tmp_path_factory.mktemp("adjoint_no_wi")
    case_path = _copy_case_folder(base_tmp, tiny_folder, "TINY_NO_WI")
    case_options = _adjoint_options(options, well_index_from_perm=False)
    _, gradient = _run_case(case_path, case_options)
    return gradient.loc[GRADIENT_STEP, (GRADIENT_TARGET, "permx")]


@pytest.fixture(scope="module")
def gradient_with_wi_chain_rule(tmp_path_factory, options, tiny_folder):
    """Adjoint gradient with the default ("auto") well-index handling."""
    base_tmp = tmp_path_factory.mktemp("adjoint_wi")
    case_path = _copy_case_folder(base_tmp, tiny_folder, "TINY_WI")
    _, gradient = _run_case(case_path, _adjoint_options(options))
    return gradient.loc[GRADIENT_STEP, (GRADIENT_TARGET, "permx")]


@pytest.fixture(scope="module")
def finite_difference_gradient(tmp_path_factory, options, tiny_folder):
    """Central-difference dWOPR:PRO2/dPERMX at the cells in `FD_CELLS`."""
    base_tmp = tmp_path_factory.mktemp("finite_difference")
    case_path = _copy_case_folder(base_tmp, tiny_folder, "TINY_FD")

    permx = np.load(case_path / "PERMX.npy")
    inputs = []
    for cell in FD_CELLS:
        for sign in (1.0, -1.0):
            perturbed = permx.copy()
            perturbed[cell] *= 1.0 + sign * FD_EPS
            inputs.append({"log_permx": np.log(perturbed)})

    simulator = JutulDarcy({**options, "parallel": min(len(inputs), 6)})
    with _working_directory(case_path):
        results = simulator(inputs)

    values = np.array([r.loc[GRADIENT_STEP, GRADIENT_TARGET] for r in results])
    plus, minus = values[0::2], values[1::2]
    return dict(zip(FD_CELLS, (plus - minus) / (2 * FD_EPS * permx[FD_CELLS])))


def test_wi_chain_rule_only_changes_perforated_cells(
    gradient_with_wi_chain_rule, gradient_without_wi_chain_rule
):
    """The extra term touches perforated cells and leaves the rest alone."""
    untouched = [
        c for c in range(gradient_with_wi_chain_rule.size)
        if c not in PERFORATED_CELLS
    ]
    np.testing.assert_allclose(
        gradient_with_wi_chain_rule[untouched],
        gradient_without_wi_chain_rule[untouched],
    )
    assert not np.allclose(
        gradient_with_wi_chain_rule[PERFORATED_CELLS],
        gradient_without_wi_chain_rule[PERFORATED_CELLS],
    ), "Well-index chain rule made no difference at any perforated cell"


def test_wi_chain_rule_recovers_finite_difference_gradient(
    gradient_with_wi_chain_rule, gradient_without_wi_chain_rule,
    finite_difference_gradient
):
    """
    Perforated cells only agree with finite differences once the well-index
    chain rule is included; unperforated cells agree either way.
    """
    for cell, reference in finite_difference_gradient.items():
        np.testing.assert_allclose(
            gradient_with_wi_chain_rule[cell], reference, rtol=0.05,
            err_msg=f"Adjoint and finite-difference gradients differ at cell {cell}",
        )

    for cell in FD_CELLS:
        if cell in PERFORATED_CELLS:
            # Dropping the term leaves a gradient that is wrong by far more
            # than the finite-difference truncation error.
            assert not np.isclose(
                gradient_without_wi_chain_rule[cell],
                finite_difference_gradient[cell], rtol=0.5,
            ), f"Cell {cell} is perforated but shows no missing well-index term"
        else:
            np.testing.assert_allclose(
                gradient_without_wi_chain_rule[cell],
                finite_difference_gradient[cell], rtol=0.05,
            )


def test_explicit_compdat_well_index_gets_no_chain_term(
    tmp_path, options, tiny_folder
):
    """
    A deck that states the connection factor explicitly has a well index that
    does not depend on permeability, so the gradient must be left alone.
    """
    case_path = _copy_case_folder(tmp_path, tiny_folder, "TINY_FIXED_WI")
    include = case_path / "include"
    shutil.copy(include / "Schdl_fixed_wi.sch", include / "Schdl.sch")

    log_permx = np.log(np.load(case_path / "PERMX.npy"))
    with _working_directory(case_path):
        _, auto = JutulDarcy(_adjoint_options(options))({"log_permx": log_permx})
        _, off = JutulDarcy(
            _adjoint_options(options, well_index_from_perm=False)
        )({"log_permx": log_permx})

    np.testing.assert_allclose(
        auto.loc[GRADIENT_STEP, (GRADIENT_TARGET, "permx")],
        off.loc[GRADIENT_STEP, (GRADIENT_TARGET, "permx")],
    )


# --------------------------------------------------------------------------- #
# Copied permeability directions
# --------------------------------------------------------------------------- #
def test_copied_perm_directions_are_auto_detected(tmp_path, options, tiny_folder):
    """
    `RUNFILE.mako` COPYs PERMX into PERMY and PERMZ, so the default "auto"
    setting must reach the same gradient as declaring `perm_copied=True`, and
    both must differ from the PERMX-only partial derivative.
    """
    case_path = _copy_case_folder(tmp_path, tiny_folder, "TINY_AUTO_COPY")
    log_permx = np.log(np.load(case_path / "PERMX.npy"))

    def gradient(**extra):
        opts = _adjoint_options(options)
        opts.pop("perm_copied")
        opts.update(extra)
        with _working_directory(case_path):
            _, grad = JutulDarcy(opts)({"log_permx": log_permx})
        return grad.loc[GRADIENT_STEP, (GRADIENT_TARGET, "permx")]

    auto = gradient()                          # default: "auto"
    declared = gradient(perm_copied=True)
    partial = gradient(perm_copied=False)

    np.testing.assert_allclose(auto, declared)
    assert not np.allclose(auto, partial), (
        "PERMX-only partial should differ from the total derivative"
    )


# --------------------------------------------------------------------------- #
# Copy detection (deck-level, no simulation)
# --------------------------------------------------------------------------- #
#: The COPY block as it appears in the shipped `RUNFILE.mako`.
TINY_COPY_BLOCK = "COPY\n 'PERMX'  'PERMY'  /\n 'PERMX'  'PERMZ' /\n/\n"

#: An independent, non-constant PERMZ used to check per-direction detection.
INDEPENDENT_PERMZ = 10.0 + 40.0 * np.abs(np.sin(np.arange(200) * 1.7))


def _detect_for_deck(case_path: Path, copy_block: str):
    """Set up a case whose COPY section is replaced, and run copy detection."""
    from juliacall import Main as julia
    from subsurface.multphaseflow.jutul_darcy import _detect_copied_perm_axes

    julia.seval("using JutulDarcy, Jutul")
    template = case_path / "RUNFILE.mako"
    original = template.read_text()
    assert TINY_COPY_BLOCK in original, "COPY block not found in RUNFILE.mako"
    template.write_text(original.replace(TINY_COPY_BLOCK, copy_block, 1))

    # The deck INCLUDEs '../include/...', so it has to be rendered one level
    # below the case root -- exactly what `run_fwd_sim` does with `En_<n>`.
    run_dir = case_path / "En_detect"
    run_dir.mkdir()
    log_permx = np.log(np.load(case_path / "PERMX.npy"))
    datafile = JutulDarcy.render_makofile(str(template), str(run_dir),
                                          {"log_permx": log_permx})
    with _working_directory(run_dir):
        case = julia.seval(f'setup_case_from_data_file("{datafile}")')
        actnum = JutulDarcy._extract_actnum(case)
        spec = _detect_copied_perm_axes(case, actnum, "auto", julia)
    return spec


@pytest.mark.parametrize(
    "name, copy_block, expected_include, expected_ratio",
    [
        (
            "copy_to_both",
            TINY_COPY_BLOCK,
            [True, True, True], [1.0, 1.0, 1.0],
        ),
        (
            "copy_to_permy_only",
            "COPY\n 'PERMX'  'PERMY'  /\n/\n\nPERMZ\n"
            + "\n".join(f"{v:.4f}" for v in INDEPENDENT_PERMZ) + "\n/\n",
            [True, True, False], [1.0, 1.0, 0.0],
        ),
        (
            "copy_then_multiply",
            TINY_COPY_BLOCK + "\nMULTIPLY\n 'PERMZ' 0.1 /\n/\n",
            [True, True, True], [1.0, 1.0, 0.1],
        ),
    ],
    ids=["copy_to_both", "copy_to_permy_only", "copy_then_multiply"],
)
def test_copy_detection_per_direction(
    tmp_path, tiny_folder, name, copy_block, expected_include, expected_ratio
):
    """
    Each permeability direction is judged separately, so a deck that copies
    PERMX into PERMY only -- or scales PERMZ after copying -- is described
    exactly rather than collapsed to a single boolean.
    """
    case_path = _copy_case_folder(tmp_path, tiny_folder, f"TINY_{name}")
    spec = _detect_for_deck(case_path, copy_block)

    np.testing.assert_array_equal(spec.include, expected_include)
    np.testing.assert_allclose(spec.ratio, expected_ratio, rtol=1e-9)
