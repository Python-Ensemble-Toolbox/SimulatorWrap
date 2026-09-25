"""
Simulator wrapper for the JutulDarcy simulator.

This module provides a Python interface (:class:`JutulDarcy`) for running
JutulDarcy reservoir simulations from a single configuration dictionary.
It supports:

* Single or ensemble forward simulations (parallel via :mod:`p_tqdm`).
* Input through either a static ``.DATA`` file or a Mako-templated ``.mako``
  file rendered per ensemble member.
* Flexible result extraction (field and per-well summary keywords) with
  ``list`` / ``dict`` / ``DataFrame`` output formats.
* Adjoint sensitivity computation for well-based objectives with respect to
  reservoir parameters (porosity, permeability, optionally log-scaled or
  copied across PERMX/Y/Z). Permeability gradients account for the two ways a
  deck makes other quantities depend on the permeability field: a COMPDAT
  connection factor left defaulted (well-index chain rule) and PERMY/PERMZ
  produced by ``COPY`` from PERMX. Both are detected from the deck by default.

The wrapper communicates with Julia via :mod:`juliacall`. Heavy Julia state
is created lazily inside worker processes so the class can be pickled for
multiprocessing.
"""

import hashlib
import os
import shutil
import warnings
import datetime as dt
import numpy as np
import pandas as pd

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from mako.template import Template
from p_tqdm import p_map
from tqdm import tqdm

__author__ = ["Mathias Methlie Nilsen"] # With help from "Claude Opus 4.7"
__all__ = ["JutulDarcy"]


# ============================================================================ #
# Environment configuration
# ============================================================================ #
# These environment variables must be set BEFORE juliacall is imported. We use
# `setdefault` so users can still override them externally.
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
os.environ.setdefault("PYTHON_JULIACALL_OPTLEVEL", "3")

# Silence the noisy "juliacall module already imported" warning that appears
# in worker processes when Julia is re-imported.
warnings.filterwarnings("ignore", message=".*juliacall module already imported.*")


# ============================================================================ #
# Module-level constants
# ============================================================================ #
SECONDS_PER_DAY: int = 86_400

#: Shared progress-bar styling used by both forward and adjoint loops.
PBAR_OPTS: dict[str, Any] = {
    "ncols": 110,
    "colour": "#285475",
    "bar_format": (
        "{desc}: {percentage:3.0f}% [{bar}] {n_fmt}/{total_fmt} "
        "│ ⏱ {elapsed}<{remaining} │ {rate_fmt}"
    ),
    "ascii": "-◼",
}

#: Summary keywords whose values are cumulative (totals), as opposed to rates.
CUMULATIVE_KEYS: frozenset[str] = frozenset({
    "FOPT", "FGPT", "FWPT", "FWLT", "FWIT",
    "WOPT", "WGPT", "WWPT", "WLPT",
})

VALID_OUTPUT_FORMATS: frozenset[str] = frozenset({"list", "dict", "dataframe"})
VALID_ADJOINT_MODES: frozenset[str]  = frozenset({"sensitivities", "optimization"})
VALID_REPORT_TYPES: frozenset[str]   = frozenset({"days", "dates"})

#: Map summary keyword -> (phase string, is_rate). Cumulative totals
#: (``WxPT``) are marked ``is_rate=False``; instantaneous rates (``WxPR``,
#: ``WxIR``) are marked ``is_rate=True``.
PHASE_MAP: dict[str, tuple[str, bool]] = {
    "WOPT": ("oil", False),    "WGPT": ("gas", False),
    "WWPT": ("water", False),  "WLPT": ("liquid", False),
    "WOPR": ("oil", True),     "WGPR": ("gas", True),
    "WWPR": ("water", True),   "WWIR": ("water", True),
    "WLPR": ("liquid", True),
}

#: Map a phase name to the corresponding JutulDarcy rate-target type name.
RATE_ID_MAP: dict[str, str] = {
    "mass":   "TotalSurfaceMassRate",
    "liquid": "SurfaceLiquidRateTarget",
    "water":  "SurfaceWaterRateTarget",
    "oil":    "SurfaceOilRateTarget",
    "gas":    "SurfaceGasRateTarget",
    "rate":   "TotalRateTarget",
}

#: Display units for known summary keywords (metric system).
UNIT_MAP: dict[str, str] = {
    "PORO": "", "PERMX": "mD", "PERMY": "mD", "PERMZ": "mD",
    "FOPT": "Sm3", "FGPT": "Sm3", "FWPT": "Sm3", "FWLT": "Sm3", "FWIT": "Sm3",
    "FOPR": "Sm3/day", "FGPR": "Sm3/day", "FWPR": "Sm3/day",
    "FLPR": "Sm3/day", "FWIR": "Sm3/day",
    "WOPR": "Sm3/day", "WGPR": "Sm3/day", "WWPR": "Sm3/day",
    "WLPR": "Sm3/day", "WWIR": "Sm3/day",
}

#: Permeability keywords in canonical (x, y, z) order.
PERM_KEYS: tuple[str, ...] = ("PERMX", "PERMY", "PERMZ")

#: Map lower-case permeability parameter substring -> axis index in PERM_KEYS.
PERM_INDEX: dict[str, int] = {"permx": 0, "permy": 1, "permz": 2}

#: Accepted values for the ``well_index_from_perm`` option.
VALID_WI_MODES: frozenset[Any] = frozenset({"auto", True, False})

#: Accepted values for the ``perm_copied`` option.
VALID_PERM_COPY_MODES: frozenset[Any] = frozenset({"auto", True, False})

#: Relative tolerance for deciding that PERMY/PERMZ is a constant multiple of
#: PERMX. A deck ``COPY`` is applied verbatim, so a plain copy matches exactly;
#: the tolerance only absorbs the rounding of a following ``MULTIPLY``.
PERM_COPY_RTOL: float = 1e-10

#: Relative tolerance used to recognise a well index that JutulDarcy itself
#: computed from the permeability field. A defaulted COMPDAT connection factor
#: is bit-for-bit what :func:`compute_peaceman_index` returns, so anything that
#: survives this comparison came from the Peaceman formula rather than the deck.
WI_PEACEMAN_RTOL: float = 1e-10

#: Julia helpers that JutulDarcy does not expose itself. Defined once per Julia
#: session by :func:`_ensure_julia_helpers`.
#:
#: ``subsurface_sensitivities_with_wells``
#:     ``JutulDarcy.reservoir_sensitivities`` narrows the adjoint result to the
#:     ``:Reservoir`` submodel and throws the rest away, which loses
#:     ``dJ/dWellIndices``. This repeats its post-processing but also returns the
#:     untouched per-model sensitivities, so one adjoint solve serves both.
#:
#: ``subsurface_wi_perm_jacobian``
#:     Differentiates JutulDarcy's own Peaceman routine, so ``dWI/dK`` follows
#:     whatever formula (and defaults handling) the installed version uses.
#:
#: ``subsurface_adjoint_packed`` / ``subsurface_adjoint_storage``
#:     The per-member work that `solve_adjoint_sensitivities` repeats for every
#:     objective: expanding the result to ministeps, and allocating the two
#:     adjoint simulators. Hoisting them out is worth ~20% of a sweep.
#:
#: ``subsurface_packed_step_times``
#:     Substep times as the adjoint solver itself sees them, used to decide how
#:     far back a given objective actually has to sweep.
#:
#: ``subsurface_sensitivities_reuse``
#:     One adjoint sweep against pre-built storage, stopping at `n_steps`. An
#:     objective that only fires at report step k has zero adjoint contribution
#:     from every later step, so sweeping them is wasted work.
_JULIA_HELPERS: str = """
function subsurface_sensitivities_with_wells(case, result, obj)
    sens = Jutul.solve_adjoint_sensitivities(case, result, obj)
    rmodel = JutulDarcy.reservoir_model(case.model)
    rsens = haskey(sens, :Reservoir) ? sens[:Reservoir] : sens
    grad = Jutul.data_domain_to_parameters_gradient(rmodel, rsens)
    for (k, pdef) in pairs(Jutul.get_parameters(rmodel))
        grad[k, Jutul.associated_entity(pdef)] = rsens[k]
    end
    return (reservoir = grad, models = sens)
end

function subsurface_wi_perm_jacobian(case, rtol)
    out = Dict{Symbol, Any}()
    model = case.model
    rdomain = JutulDarcy.reservoir_domain(model)
    ncells = Jutul.number_of_cells(rdomain)
    model isa Jutul.MultiModel || return (ncells = ncells, wells = out)
    gdim = Jutul.dim(Jutul.physical_representation(rdomain))
    p = JutulDarcy.Perforations()
    for (name, m) in pairs(model.models)
        JutulDarcy.model_or_domain_is_well(m) || continue
        dd = m.data_domain
        WI    = dd[:well_index, p]
        dims  = dd[:cell_dims, p]
        perm  = dd[:permeability, p]
        ntg   = dd[:net_to_gross, p]
        dir   = dd[:perforation_direction, p]
        skin  = dd[:skin, p]
        Kh    = dd[:Kh, p]
        rad   = dd[:perforation_radius, p]
        drain = dd[:drainage_radius, p]
        cells = Jutul.physical_representation(m.domain).perforations.reservoir
        n = length(WI)
        nrow = perm isa AbstractVector ? 1 : size(perm, 1)
        J = zeros(nrow, n)
        from_perm = falses(n)
        for i in 1:n
            K = Float64.(perm isa AbstractVector ? [perm[i]] : collect(perm[:, i]))
            peaceman = kk -> JutulDarcy.compute_peaceman_index(
                dims[i], Jutul.expand_perm(kk, gdim), rad[i], dir[i];
                skin = skin[i],
                Kh = Kh[i],
                net_to_gross = ntg[i],
                drainage_radius = drain[i],
                check = false
            )
            from_perm[i] = isapprox(WI[i], peaceman(K), rtol = rtol)
            J[:, i] = JutulDarcy.ForwardDiff.gradient(peaceman, K)
        end
        out[name] = (
            cells = collect(cells),
            dWI_dK = J,
            from_perm = collect(from_perm)
        )
    end
    return (ncells = ncells, wells = out)
end

function subsurface_adjoint_packed(case, result)
    simresult = hasproperty(result, :result) ? result.result : result
    states, dt, step_ix = Jutul.expand_to_ministeps(simresult)
    forces = case.forces
    if forces isa Vector
        forces = forces[step_ix]
    end
    return (states = states, dt = dt, forces = forces)
end

function subsurface_adjoint_storage(case)
    return Jutul.setup_adjoint_storage(case.model;
        state0 = case.state0,
        parameters = case.parameters
    )
end

function subsurface_packed_step_times(packed)
    ps = Jutul.AdjointPackedResult(packed.states, packed.dt, packed.forces)
    return [ps[i].step_info[:time] for i in 1:length(ps)]
end

function subsurface_sensitivities_reuse(case, storage, packed, obj, n_steps)
    # Objective sparsity is cached on the storage after the first solve, so it
    # has to be dropped between objectives -- they touch different wells.
    osp = storage.objective_sparsity
    if !isnothing(osp)
        osp[:forward] = nothing
        osp[:parameter] = nothing
    end

    # `setup_adjoint_storage` builds the linear solver as a per-call default, so
    # the original code gave every objective a fresh GenericKrylov -- fresh
    # preconditioner and fresh scaling state. Reusing one across objectives
    # changes the iterates and moves the gradients around inside the solver
    # tolerance, so rebuild it here. It is cheap next to the two simulators the
    # storage holds, which is what we are actually hoisting.
    storage.forward_config[:linear_solver] =
        Jutul.select_linear_solver(case.model, mode = :adjoint, rtol = 1e-6)

    @. storage.dx = 0
    @. storage.rhs = 0
    @. storage.lagrange = 0
    @. storage.lagrange_buffer = 0

    n = min(n_steps, length(packed.states))
    forces = packed.forces isa Vector ? packed.forces[1:n] : packed.forces
    pmodel = storage.parameter.model
    dG = zeros(Jutul.number_of_degrees_of_freedom(pmodel))
    Jutul.solve_adjoint_sensitivities!(
        dG, storage,
        packed.states[1:n], case.state0, packed.dt[1:n], obj;
        forces = forces
    )

    sens = Jutul.store_sensitivities(pmodel, dG, storage.parameter_map)
    rmodel = JutulDarcy.reservoir_model(case.model)
    rsens = haskey(sens, :Reservoir) ? sens[:Reservoir] : sens
    grad = Jutul.data_domain_to_parameters_gradient(rmodel, rsens)
    for (k, pdef) in pairs(Jutul.get_parameters(rmodel))
        grad[k, Jutul.associated_entity(pdef)] = rsens[k]
    end
    return (reservoir = grad, models = sens)
end
"""


# ============================================================================ #
# Configuration dataclasses
# ============================================================================ #
@dataclass
class PermCopySpec:
    """
    Which permeability directions are slaved to PERMX, and by what factor.

    A deck that writes only PERMX and then ``COPY``s it into PERMY/PERMZ has a
    single permeability degree of freedom, so the derivative w.r.t. that
    control is the sum over all three directions rather than the PERMX partial
    alone. ``ratio`` carries the constant factor of any ``MULTIPLY`` applied
    after the copy (1.0 for a plain copy).

    Attributes
    ----------
    include : np.ndarray
        Boolean mask over permeability directions; ``include[0]`` (PERMX, the
        master) is always True.
    ratio : np.ndarray
        ``PERM<axis> / PERMX`` for included directions, 0.0 otherwise.
    """
    include: np.ndarray
    ratio: np.ndarray


@dataclass
class AdjointObjective:
    """
    One adjoint objective specification (one well + phase).

    Attributes
    ----------
    wellID : str
        Name of the well associated with this objective.
    phase : str
        Phase string used to pick the rate target (see :data:`RATE_ID_MAP`).
    is_rate : bool
        True for instantaneous rate objectives, False for cumulative totals.
    parameters : list[str]
        Parameters w.r.t. which the gradient should be evaluated, e.g.
        ``["PORO", "PERMX"]`` (case-insensitive; ``"log"`` enables log-scaling).
    steps : Any
        Either the string ``"all"`` (use the wrapper's global report points)
        or an explicit list of ``int`` days or :class:`datetime.datetime`
        objects at which to evaluate the objective.
    """
    wellID: str
    phase: str
    is_rate: bool
    parameters: list[str]
    steps: Any  # 'all' | list[int] | list[datetime]


# ============================================================================ #
# Helper functions (pure / stateless)
# ============================================================================ #
def get_metric_unit(key: str) -> str:
    """
    Return the metric-system unit string for a summary keyword.

    Parameters
    ----------
    key : str
        Summary keyword (case-insensitive), e.g. ``"FOPT"``.

    Returns
    -------
    str
        Unit string from :data:`UNIT_MAP`, or ``"Unknown"`` if not found.
    """
    return UNIT_MAP.get(key.upper(), "Unknown")


def _process_datatype_info(datatypes: Iterable[str]) -> list[str]:
    """
    Expand grouped datatype specs into one entry per well.

    A spec like ``"WOPR:W1:W2"`` is expanded to ``["WOPR:W1", "WOPR:W2"]``.
    Specs without a colon (field-level) are passed through unchanged.

    Parameters
    ----------
    datatypes : iterable of str
        User-supplied datatype identifiers.

    Returns
    -------
    list of str
        Normalised, one-well-per-entry list.
    """
    out: list[str] = []
    for d in datatypes:
        if ":" in d:
            base, *wells = d.split(":")
            out.extend(f"{base}:{w}" for w in wells)
        else:
            out.append(d)
    return out


def _process_adjoint_info(adjoint_info: dict) -> dict[str, AdjointObjective]:
    """
    Normalise the ``adjoints`` config dict into :class:`AdjointObjective` items.

    Each input entry may specify a single well or a list of wells; the output
    contains one :class:`AdjointObjective` per (datatype, well) pair, keyed by
    ``"<datatype>:<well>"``.

    Parameters
    ----------
    adjoint_info : dict
        Raw mapping ``{datatype: {"wellID": ..., "parameters": ..., "steps": ...}}``.

    Returns
    -------
    dict[str, AdjointObjective]
        Flattened objective specifications.
    """
    info: dict[str, AdjointObjective] = {}
    for dataID, spec in adjoint_info.items():
        # Normalise scalar values to lists for uniform iteration.
        wells = spec["wellID"]
        if isinstance(wells, str):
            wells = [wells]
        params = spec["parameters"]
        if isinstance(params, str):
            params = [params]

        phase, is_rate = PHASE_MAP[dataID]
        for w in wells:
            info[f"{dataID}:{w}"] = AdjointObjective(
                wellID=w, phase=phase, is_rate=is_rate,
                parameters=list(params), steps=spec["steps"],
            )
    return info


def _active_to_full_grid(vec: np.ndarray, actnum_vec: np.ndarray,
                         fill_value: float = 0.0) -> np.ndarray:
    """
    Embed an active-cell vector into the full grid layout.

    Parameters
    ----------
    vec : np.ndarray
        Either a vector defined only on active cells (length ``actnum_vec.sum()``)
        or already on the full grid (length ``len(actnum_vec)``).
    actnum_vec : np.ndarray
        Flat ACTNUM array (1 = active, 0 = inactive), Fortran-ordered.
    fill_value : float, optional
        Value placed at inactive cells. Defaults to ``0.0``.

    Returns
    -------
    np.ndarray
        Full-grid vector of length ``len(actnum_vec)`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``vec`` matches neither the active nor the full grid size.
    """
    n_active = int(actnum_vec.sum())
    if len(vec) == n_active:
        full = np.full(actnum_vec.shape, fill_value, dtype=np.float64, order="F")
        full[actnum_vec == 1] = vec
        return full
    if len(vec) == len(actnum_vec):
        return np.asarray(vec, dtype=np.float64)
    raise ValueError("Parameter length does not match number of active cells")


@contextmanager
def _chdir(path: str | os.PathLike):
    """
    Context manager that temporarily changes the working directory.

    The previous working directory is always restored, including when the
    wrapped block raises an exception.

    Parameters
    ----------
    path : str or path-like
        Directory to switch into.

    Yields
    ------
    None
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _suppress_julia(julia, code: str):
    """
    Evaluate a Julia expression with ``stdout`` and ``stderr`` redirected to devnull.

    Used to silence verbose JutulDarcy setup/solver messages.

    Parameters
    ----------
    julia : juliacall.Main
        The Julia main module reference.
    code : str
        A single Julia expression (will be wrapped, not multi-line statements).

    Returns
    -------
    Any
        Result of the inner Julia expression.
    """
    return julia.seval(f"""
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                {code}
            end
        end
    """)


def _get_mapping_value(obj, key_name: str, julia):
    """
    Look up ``key_name`` in a Julia mapping using both Symbol and string keys.

    JutulDarcy mappings sometimes use ``Symbol`` keys and sometimes plain
    strings; this helper tries both so callers don't have to.

    Parameters
    ----------
    obj : Any
        A Julia mapping-like object supporting ``haskey`` / ``getindex``.
    key_name : str
        Candidate key as a Python string.
    julia : juliacall.Main
        Julia main module (used to construct ``Symbol``).

    Returns
    -------
    Any or None
        The value if found; ``None`` otherwise.
    """
    for candidate in (julia.Symbol(key_name), key_name):
        try:
            if julia.haskey(obj, candidate):
                return obj[candidate]
        except Exception:
            # `haskey` may not be defined for every object type encountered.
            continue
    return None


def _extract_key_value(root_object, keys, julia):
    """
    Breadth-first search for the first matching key in nested Julia mappings.

    Useful because JutulDarcy gradient structures can nest gradients several
    layers deep (e.g. ``grad[:model][:reservoir][:porosity]``).

    Parameters
    ----------
    root_object : Any or list
        Starting Julia object(s) to search.
    keys : str or list of str
        Candidate key name(s). The first key found anywhere in the tree wins.
    julia : juliacall.Main
        Julia main module.

    Returns
    -------
    Any or None
        Value associated with the first matching key, or ``None``.
    """
    if not isinstance(keys, list):
        keys = [keys]
    if not isinstance(root_object, list):
        root_object = [root_object]

    queue = deque(root_object)
    visited: set[int] = set()  # avoid revisiting the same Julia object

    while queue:
        cur = queue.popleft()
        oid = id(cur)
        if oid in visited:
            continue
        visited.add(oid)

        # Try to match at the current level first.
        for k in keys:
            val = _get_mapping_value(cur, k, julia)
            if val is not None:
                return val

        # Otherwise, enqueue children for further exploration.
        try:
            for ck in list(julia.keys(cur)):
                try:
                    queue.append(cur[ck])
                except Exception:
                    continue
        except Exception:
            # `cur` is a leaf (no `keys` method); skip it.
            continue

    return None


def _detect_copied_perm_axes(case, actnum: np.ndarray, perm_copied: Any,
                             julia) -> PermCopySpec | None:
    """
    Work out which permeability directions move with PERMX.

    ``COPY``/``MULTIPLY`` are applied while the deck is parsed and leave no
    record behind, so the copy is recovered numerically: a direction counts as
    slaved when its array is a constant multiple of PERMX over every active
    cell.

    Parameters
    ----------
    case : Any
        Julia case object.
    actnum : np.ndarray
        Flat ACTNUM vector (1/0).
    perm_copied : {"auto", True, False}
        ``"auto"`` detects; ``True`` forces all three directions in at ratio
        1.0 (the historical behaviour); ``False`` disables summation.
    julia : juliacall.Main
        Julia main module.

    Returns
    -------
    PermCopySpec or None
        ``None`` when the gradient should stay a per-direction partial.

    Warns
    -----
    UserWarning
        If PERMX is uniform, in which case the test cannot discriminate.

    Notes
    -----
    Directions are judged one at a time, so copying PERMX into PERMY while
    specifying PERMZ independently is handled correctly.

    A copy applied to only part of the grid (a ``BOX``) is deliberately *not*
    detected: the ratio is then non-constant, and a spatially varying ratio
    cannot be told apart from two independently specified fields.
    """
    if perm_copied is False:
        return None
    n = len(PERM_KEYS)
    if perm_copied is True:
        return PermCopySpec(np.ones(n, dtype=bool), np.ones(n, dtype=np.float64))

    include = np.zeros(n, dtype=bool)
    ratio = np.zeros(n, dtype=np.float64)
    include[0], ratio[0] = True, 1.0

    grid = case.input_data["GRID"]
    active = actnum == 1

    def _column(key: str) -> np.ndarray | None:
        if not julia.haskey(grid, key):
            return None
        return np.asarray(grid[key], dtype=np.float64).flatten(order="F")[active]

    master = _column(PERM_KEYS[0])
    if master is None or not np.any(master):
        # Nothing to compare against, so no direction can be shown to be a copy.
        return None

    # Proportionality carries no information about a uniform PERMX: every
    # constant field is a multiple of it, so a genuinely independent PERMZ
    # would be indistinguishable from a copy. Stay with the partial derivative
    # and let the user settle it.
    if np.ptp(master) <= PERM_COPY_RTOL * abs(float(np.mean(master))):
        warnings.warn(
            "PERMX is uniform, so a COPY into PERMY/PERMZ cannot be told apart "
            "from independently specified fields. Falling back to per-direction "
            "partial derivatives; set 'perm_copied' to True or False to choose "
            "explicitly.",
            UserWarning,
            stacklevel=2,
        )
        return None

    denom = float(master @ master)
    for axis in range(1, n):
        other = _column(PERM_KEYS[axis])
        if other is None:
            continue
        # Least-squares ratio, then verified pointwise so that a merely
        # correlated field is not mistaken for a copy.
        r = float(master @ other) / denom
        if r != 0.0 and np.allclose(other, r * master, rtol=PERM_COPY_RTOL, atol=0.0):
            include[axis], ratio[axis] = True, r

    if not include[1:].any():
        # No direction is slaved to PERMX: the three fields are independent, so
        # each parameter must keep its own partial derivative. Returning a spec
        # would collapse them onto the master and silently give dJ/dPERMX for
        # permy and permz too.
        return None
    return PermCopySpec(include, ratio)


def _ensure_julia_helpers(julia) -> None:
    """
    Define :data:`_JULIA_HELPERS` in the Julia session if not already present.

    Julia state persists for the lifetime of a worker process, so this is a
    no-op on every call after the first.

    The marker is derived from the helper source, so a session that already
    holds an older definition redefines rather than silently keeping it.

    Parameters
    ----------
    julia : juliacall.Main
        Julia main module.
    """
    digest = hashlib.md5(_JULIA_HELPERS.encode()).hexdigest()[:12]
    marker = f"subsurface_helpers_{digest}"
    if julia.seval(f"@isdefined({marker})"):
        return
    julia.seval(_JULIA_HELPERS)
    julia.seval(f"{marker} = true")


def _well_index_perm_jacobian(case, julia,
                              well_index_from_perm: Any = "auto"
                              ) -> tuple[dict[str, dict], int]:
    """
    Differentiate every perforation's well index w.r.t. cell permeability.

    A ``.DATA`` deck that leaves the COMPDAT connection-transmissibility factor
    defaulted makes the well index a *function* of the permeability field via
    the Peaceman formula. JutulDarcy evaluates that function once during case
    setup and then treats ``WellIndices`` as an independent parameter, so its
    permeability gradient is a partial derivative that misses
    ``(dJ/dWI)·(dWI/dK)``. This returns the ``dWI/dK`` half of that term.

    Parameters
    ----------
    case : Any
        Julia case object.
    julia : juliacall.Main
        Julia main module.
    well_index_from_perm : {"auto", True, False}, optional
        Which perforations to treat as permeability-derived. ``"auto"``
        (default) keeps the ones whose stored well index matches a fresh
        Peaceman evaluation, i.e. exactly the defaulted COMPDAT entries.
        ``True`` forces every perforation, ``False`` selects none.

    Returns
    -------
    dict[str, dict]
        ``{well_name: {"cells": 1-based reservoir cell index per kept
        perforation, "keep": boolean mask over all of the well's perforations,
        "dWI_dK": (n_axis, n_kept) array}}``. Wells with no kept perforation
        are omitted.
    int
        Number of cells in the reservoir domain, i.e. the column count of the
        reservoir permeability gradient. Zero when no jacobian was built.
    """
    if well_index_from_perm is False:
        return {}, 0

    _ensure_julia_helpers(julia)
    jac = julia.subsurface_wi_perm_jacobian(case, WI_PEACEMAN_RTOL)
    ncells = int(jac.ncells)
    wells = jac.wells
    out: dict[str, dict] = {}
    for name in julia.keys(wells):
        entry = wells[name]
        keep = (
            np.ones(len(entry.cells), dtype=bool)
            if well_index_from_perm is True
            else np.asarray(entry.from_perm, dtype=bool)
        )
        if not keep.any():
            continue
        out[str(name)] = {
            "cells": np.asarray(entry.cells, dtype=np.int64)[keep],
            "keep": keep,
            "dWI_dK": np.asarray(entry.dWI_dK, dtype=np.float64)[:, keep],
        }
    return out, ncells


def _well_index_chain_term(wi_jacobian: dict[str, dict], wi_gradients,
                           ncells: int, julia) -> np.ndarray:
    """
    Assemble ``sum_perf (dJ/dWI)·(dWI/dK)`` on the active-cell permeability grid.

    Parameters
    ----------
    wi_jacobian : dict[str, dict]
        Output of :func:`_well_index_perm_jacobian`; must be non-empty.
    wi_gradients : Any
        Julia mapping from well name to that well's parameter sensitivities;
        each entry is expected to hold a ``WellIndices`` vector.
    ncells : int
        Number of cells in the reservoir domain.
    julia : juliacall.Main
        Julia main module.

    Returns
    -------
    np.ndarray
        ``(n_axis, ncells)`` array, zero everywhere except at perforated cells.
        ``n_axis`` follows the permeability layout of the reservoir domain.

    Raises
    ------
    ValueError
        If a well's ``WellIndices`` sensitivity cannot be located.
    """
    n_axis = next(iter(wi_jacobian.values()))["dWI_dK"].shape[0]
    term = np.zeros((n_axis, ncells), dtype=np.float64)
    for well, entry in wi_jacobian.items():
        well_sens = _get_mapping_value(wi_gradients, well, julia)
        wi_grad = (
            None if well_sens is None
            else _get_mapping_value(well_sens, "WellIndices", julia)
        )
        if wi_grad is None:
            raise ValueError(
                f"Could not find the WellIndices sensitivity for well '{well}'; "
                "the well-index chain rule cannot be applied. Set "
                "'well_index_from_perm': False to fall back to the partial "
                "derivative."
            )
        # `wi_grad` covers every perforation of the well, so drop the ones
        # whose well index was fixed in the deck before contracting.
        dJ_dWI = np.asarray(wi_grad, dtype=np.float64)[entry["keep"]]

        # `cells` is 1-based (Julia) and may repeat when one cell holds several
        # perforations, so accumulate rather than assign.
        np.add.at(term, (slice(None), entry["cells"] - 1),
                  entry["dWI_dK"] * dJ_dWI)
    return term


def _extract_adjoint(jlgrad, jlcase, parameter: str, actnum: np.ndarray,
                     perm_copy: PermCopySpec | None, julia,
                     wi_chain_term: np.ndarray | None = None) -> np.ndarray:
    """
    Extract and post-process an adjoint gradient for a single parameter.

    Handles unit scaling (mD → SI), optional log-scaling, embedding back onto
    the full grid, and the "copied permeability" convention where PERMY/PERMZ
    are duplicates of PERMX in the input deck.

    Parameters
    ----------
    jlgrad : Any
        Julia gradient object returned by the adjoint solver.
    jlcase : Any
        Julia case object (needed to read PERMX/Y/Z values when log-scaling).
    parameter : str
        Parameter name. May contain ``"poro"``, ``"perm"``, ``"permx"``,
        ``"permy"``, ``"permz"`` (case-insensitive). Including ``"log"``
        enables log-scaling for permeability.
    actnum : np.ndarray
        Flat ACTNUM vector (1/0) for embedding into the full grid.
    perm_copy : PermCopySpec or None
        Which permeability directions are slaved to PERMX (see
        :func:`_detect_copied_perm_axes`). When given, PERMX is the master and
        the returned gradient sums the slaved directions into it. ``None``
        returns the partial derivative for the direction named in
        ``parameter``.
    julia : juliacall.Main
        Julia main module.
    wi_chain_term : np.ndarray, optional
        ``(n_axis, n_active)`` well-index chain-rule contribution from
        :func:`_well_index_chain_term`, in the same SI units as the raw Julia
        permeability gradient. Added before any scaling. Ignored for
        non-permeability parameters.

    Returns
    -------
    np.ndarray
        Full-grid gradient vector.

    Raises
    ------
    ValueError
        If the gradient could not be located, the parameter is unsupported, or
        ``wi_chain_term`` does not match the shape of the permeability gradient.
    """
    p = parameter.lower()

    # ---- Porosity -------------------------------------------------------- #
    if "poro" in p:
        grad = _extract_key_value(jlgrad, "porosity", julia)
        if grad is None:
            raise ValueError(f"Could not find porosity gradient for '{parameter}'")
        return _active_to_full_grid(np.asarray(grad), actnum)

    # ---- Permeability ---------------------------------------------------- #
    if "perm" in p:
        log_scale = "log" in p
        grad = _extract_key_value(jlgrad, "permeability", julia)
        if grad is None:
            raise ValueError(f"Could not find permeability gradient for '{parameter}'")

        full = np.asarray(grad)              # shape: (3, n_active)

        # JutulDarcy differentiates w.r.t. permeability with the well indices
        # held fixed. Where those indices were derived from the permeability
        # (defaulted COMPDAT connection factors) the total derivative needs the
        # extra (dJ/dWI)·(dWI/dK) term folded in before scaling.
        if wi_chain_term is not None:
            if wi_chain_term.shape != full.shape:
                raise ValueError(
                    f"Well-index chain term has shape {wi_chain_term.shape}, "
                    f"expected {full.shape} to match the permeability gradient"
                )
            full = full + wi_chain_term

        mdarcy = julia.seval("si_unit(:milli)*si_unit(:darcy)")  # mD → SI factor

        def _scale(adj: np.ndarray, axis: int) -> np.ndarray:
            """Apply log-scale (∂J/∂log(k) = k·∂J/∂k) or unit conversion."""
            if log_scale:
                perm = np.array(jlcase.input_data["GRID"][PERM_KEYS[axis]])
                return adj * perm.flatten(order="F")
            return adj * mdarcy

        # Copied-permeability case: PERMX is the control, so aggregate every
        # direction that moves with it.
        #
        # With PERM<i> = r_i * PERMX the total derivative is
        #     dJ/dPERMX      = sum_i r_i * dJ/dPERM<i>
        #     dJ/dlog(PERMX) = sum_i PERM<i> * dJ/dPERM<i>
        # so the log branch of `_scale` already carries r_i through PERM<i>,
        # while the unit-converted branch has to apply it explicitly.
        if perm_copy is not None:
            out = np.zeros(actnum.shape, dtype=np.float64)
            for i, (use, r) in enumerate(zip(perm_copy.include, perm_copy.ratio)):
                if not use:
                    continue
                contribution = _scale(_active_to_full_grid(full[i], actnum), i)
                out += contribution if log_scale else r * contribution
            return out

        # Otherwise pick the axis encoded in the parameter name.
        idx = next((v for k, v in PERM_INDEX.items() if k in p), None)
        if idx is None:
            raise ValueError(f"Adjoint not implemented for '{parameter}'")
        return _scale(_active_to_full_grid(full[idx], actnum), idx)

    raise ValueError(f"Adjoint not implemented for parameter '{parameter}'")


def well_QOI_objective(wellID: str, phaseID: str, time: Iterable[float],
                       step_index=None, is_rate: bool = True, julia=None):
    """
    Build per-timestep Julia QOI closures for a single well/phase.

    Each closure returns either a daily-volume contribution (``is_rate=True``,
    one time only) or the cumulative integrand over time up to a horizon
    (``is_rate=False``). The sign is flipped automatically for producers so
    objectives are positive for "good" outcomes (e.g. produced oil).

    Parameters
    ----------
    wellID : str
        Name of the well as it appears in the case.
    phaseID : str
        Phase identifier (see :data:`RATE_ID_MAP`).
    time : iterable of float
        Sequence of horizon times in seconds.
    step_index : Any, optional
        Currently unused; retained for backwards compatibility.
    is_rate : bool, optional
        Selects rate (instantaneous) vs cumulative objective formulation.
    julia : juliacall.Main, optional
        Existing Julia module; one is imported on demand if omitted.

    Returns
    -------
    list[Any]
        List of Julia closures, one per entry in ``time``.

    Raises
    ------
    ValueError
        If ``phaseID`` is not in :data:`RATE_ID_MAP`.
    """
    if julia is None:
        from juliacall import Main as julia
        julia.seval("using JutulDarcy")

    if phaseID not in RATE_ID_MAP:
        raise ValueError(f"Unknown rate type: {phaseID}")
    rate_sym = RATE_ID_MAP[phaseID]

    qois = []
    for i, sec in enumerate(time):
        if is_rate:
            # Instantaneous rate: contribute only at the exact step matching `sec`.
            obj = julia.seval(
                f"""
                function well_QOI_{i}(model, state, dt, step_info, forces)
                    if step_info[:time] != {sec}
                        return 0.0
                    end
                    ctrl = forces[:Facility].control[Symbol("{wellID}")]
                    sign = ctrl isa JutulDarcy.ProducerControl ? -1.0 : 1.0
                    rate = JutulDarcy.compute_well_qoi(model, state, forces, Symbol("{wellID}"), {rate_sym})
                    return sign * rate * si_unit(:day) # rate is in Sm3/s: convert to Sm3/day
                end
                """
            )
        else:
            # Cumulative: integrate (dt * rate) up to the horizon `sec`.
            obj = julia.seval(
                f"""
                function well_QOI_{i}(model, state, dt, step_info, forces)
                    if step_info[:time] > {sec}
                        return 0.0
                    end
                    ctrl = forces[:Facility].control[Symbol("{wellID}")]
                    sign = ctrl isa JutulDarcy.ProducerControl ? -1.0 : 1.0
                    rate = JutulDarcy.compute_well_qoi(model, state, forces, Symbol("{wellID}"), {rate_sym})
                    return sign * dt * rate
                end
                """
            )
        qois.append(obj)
    return qois


# ============================================================================ #
# Main wrapper
# ============================================================================ #
class JutulDarcy:
    """
    Python wrapper around a JutulDarcy reservoir simulation.

    The class is instantiated once with an options dictionary describing the
    simulation setup, then called like a function on either a single input
    (``dict`` or ``.DATA`` path) or a list of inputs (one per ensemble member).

    See the module docstring for the list of supported options.

    Parameters
    ----------
    options : dict
        Configuration dictionary. Recognised keys:

        - ``runfile`` : str
            Path to a ``.mako`` template or ``.DATA`` file.

        - ``reporttype`` : {"days", "dates"}
            How report points are interpreted. Default ``"days"``.

        - ``reportpoint`` : list
            Report points (numeric days or :class:`datetime.datetime`).

        - ``datatype`` : list[str]
            Summary keywords to extract. Default
            ``["FOPT", "FGPT", "FWPT", "FWIT"]``.

        - ``adjoints`` : dict
            Objective/parameter spec; enables adjoint computation.

        - ``output_format`` : {"list", "dict", "dataframe"}
            Output container type. Default ``"dataframe"``.

        - ``adjoint_pbar`` : bool
            Show a per-objective progress bar. Default ``False``.

        - ``parallel`` : int
            Number of worker processes. Default ``1``.
                    
        - ``perm_copied`` : {"auto", True, False}
            Whether PERMY/PERMZ are slaved to PERMX (deck ``COPY``), making
            permeability gradients total derivatives w.r.t. PERMX rather than
            per-direction partials. ``"auto"`` (default) detects it from the
            deck; see :func:`_detect_copied_perm_axes`.

        - ``well_index_from_perm`` : {"auto", True, False}
            Whether permeability gradients include the well-index chain rule
            for perforations whose COMPDAT connection factor is derived from
            the permeability. See :func:`_well_index_perm_jacobian`. Default
            ``"auto"``.

        - ``adjoint_mode`` : {"sensitivities", "optimization"}
            Selects the JutulDarcy gradient pathway. Default
            ``"sensitivities"``.

        - ``adjoint_reuse_storage`` : bool
            Build the adjoint storage once per member instead of once per
            objective, and stop each objective's backward sweep at its own
            evaluation point. Roughly 1.5x faster with many data points.
            Gradients shift by ~1e-4 relative: Jutul's adjoint linear solver is
            iterative and carries state on the storage, so a reused storage
            converges to slightly different points inside the solver tolerance.
            Set False for results bit-identical to the per-objective path.
            Only applies to ``adjoint_mode="sensitivities"``. Default True.

        - ``optimization_targets`` : Any
            Reserved for future use.
            
        - ``eval_adjoint_funcs`` : bool
            If True, store objective function values in ``self.adjoint_funcs``.

    Raises
    ------
    ValueError
        If ``reporttype``, ``output_format`` or ``adjoint_mode`` is invalid.
    """

    def __init__(self, options: dict):
        # ---- Runfile (template vs static deck) --------------------------- #
        runfile = options.get("runfile")
        self.makofile = runfile if runfile and runfile.endswith(".mako") else None
        self.datafile = runfile if runfile and runfile.endswith(".DATA") else None

        # ---- Report-point configuration ---------------------------------- #
        self.report_type = options.get("reporttype", "days")
        if self.report_type not in VALID_REPORT_TYPES:
            raise ValueError(
                f"Invalid reporttype '{self.report_type}'. "
                f"Must be one of {sorted(VALID_REPORT_TYPES)}"
            )
        self.report = options.get("reportpoint")
        self.index: list = [self.report_type, self.report]

        # Computed lazily during the first forward run.
        self.report_seconds: np.ndarray | None = None
        self.start_date: dt.datetime | None = None

        # ---- Datatypes to extract ---------------------------------------- #
        self.datatype = _process_datatype_info(
            options.get("datatype", ["FOPT", "FGPT", "FWPT", "FWIT"])
        )

        # ---- Adjoint configuration --------------------------------------- #
        if "adjoints" in options:
            self.compute_adjoints = True
            self.adjoint_info = _process_adjoint_info(options["adjoints"])
            self.eval_adjoint_funcs = options.get("eval_adjoint_funcs", False)
        else:
            self.compute_adjoints = False
            self.eval_adjoint_funcs = False
        self.adjoint_funcs: pd.DataFrame | None = None

        # ---- Output / execution options ---------------------------------- #
        self.output_format = options.get("output_format", "dataframe")
        if self.output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{self.output_format}'. "
                f"Must be one of {sorted(VALID_OUTPUT_FORMATS)}"
            )
        self.adjoint_pbar = options.get("adjoint_pbar", False)
        self.parallel = int(options.get("parallel", 1))
        self.perm_copied = options.get("perm_copied", "auto")
        if self.perm_copied not in VALID_PERM_COPY_MODES:
            raise ValueError(
                f"Invalid perm_copied '{self.perm_copied}'. "
                f"Must be 'auto', True or False"
            )
        self.well_index_from_perm = options.get("well_index_from_perm", "auto")
        if self.well_index_from_perm not in VALID_WI_MODES:
            raise ValueError(
                f"Invalid well_index_from_perm '{self.well_index_from_perm}'. "
                f"Must be 'auto', True or False"
            )
        self.adjoint_mode = options.get("adjoint_mode", "sensitivities")
        self.adjoint_reuse_storage = bool(
            options.get("adjoint_reuse_storage", True)
        )
        if self.adjoint_mode not in VALID_ADJOINT_MODES:
            raise ValueError(
                f"Invalid adjoint_mode '{self.adjoint_mode}'. "
                f"Must be one of {sorted(VALID_ADJOINT_MODES)}"
            )
        self.optimization_targets = options.get("optimization_targets")

        # ---- PET compatibility attributes -------------------------------- #
        # `input_dict` and `true_order` are consumed by the surrounding PET
        # framework; we keep them as direct passthroughs.
        self.input_dict = options
        self.true_order = self.index

    # ---------------------------------------------------------------- #
    # Public entry point
    # ---------------------------------------------------------------- #
    def __call__(self, inputs: list[dict] | dict | str):
        """
        Run the configured simulation for one or many ensemble members.

        Parameters
        ----------
        inputs : list[dict] or dict or str
            Either a list of inputs (one per ensemble member), a single dict
            of Mako template parameters, or a path to a ``.DATA`` deck.

        Returns
        -------
        Forward-only mode
            Single input → one result; list input → list of results. The
            result type follows ``self.output_format``.
        Adjoint mode
            Same shape as above, but each result is a ``(forward, adjoint)``
            tuple where ``adjoint`` is a :class:`pandas.DataFrame` with a
            ``(objective, parameter)`` MultiIndex on the columns.
        """
        # Normalise to a list so the parallel path is uniform.
        if isinstance(inputs, (dict, str)):
            inputs = [inputs]

        # Clean up any stale simulation folders from a previous (possibly failed) run.
        self._cleanup_simulation_folders()

        n = len(inputs)
        outputs = p_map(
            self.run_fwd_sim,
            list(inputs),
            list(range(n)),
            num_cpus=self.parallel,
            unit="sim",
            desc="Simulations",
            leave=False,
            **PBAR_OPTS,
        )

        # In adjoint mode each worker returns a (result, adjoint) tuple; split
        # them out so callers see two parallel collections rather than a list
        # of tuples.
        if self.compute_adjoints:
            results, adjoints = zip(*outputs)
            if n == 1:
                return results[0], adjoints[0]
            return list(results), list(adjoints)

        return outputs[0] if n == 1 else outputs

    # ---------------------------------------------------------------- #
    # Per-member forward simulation
    # ---------------------------------------------------------------- #
    def run_fwd_sim(self, input: dict | str, idn: int = 0,
                    delete_folder: bool = True):
        """
        Run a single forward simulation (and optionally adjoints) in isolation.

        Each call creates and operates inside its own ``En_<idn>`` folder so
        multiple workers cannot collide on intermediate files written by
        JutulDarcy.

        Parameters
        ----------
        input : dict or str
            Mako template parameters or a path to a ``.DATA`` deck.
        idn : int, optional
            Ensemble member index (used for folder naming and pbar position).
        delete_folder : bool, optional
            If True (default), remove the ``En_<idn>`` folder when done.

        Returns
        -------
        Same as :meth:`__call__` for a single member.
        """
        # Julia is imported lazily inside the worker so it picks up the
        # process-local thread/state. Importing at module scope would break
        # multiprocessing's fork/spawn semantics.
        from juliacall import Main as julia
        julia.seval("using JutulDarcy, Jutul")
        if self.compute_adjoints:
            _ensure_julia_helpers(julia)

        folder = Path(f"En_{idn}")
        folder.mkdir(exist_ok=False)

        try:
            # Stage the deck (render template or copy file).
            datafile = self._stage_input(input, folder, idn)

            # All Julia file I/O must happen relative to the deck location.
            with _chdir(folder):
                case = self._setup_case(datafile, julia)
                julia.case = case  # expose to Julia-side eval'd expressions

                units = self._detect_units(case, julia)
                actnum_vec = self._extract_actnum(case, julia)

                # Forward solve. `output_substates=True` keeps intermediate
                # states needed for adjoint reconstruction.
                jlres = julia.simulate_reservoir(
                    case, info_level=-1, output_substates=True
                )
                julia.res = jlres

                pyres = self.extract_datatypes(jlres, case, units, julia)
                output = self._format_output(pyres)

                if self.compute_adjoints:
                    adjoints = self._compute_adjoints(
                        case, jlres, pyres, units, actnum_vec, idn, julia
                    )
                    return output, adjoints

                return output
        finally:
            # Always clean up, even on failure, to keep the workspace tidy.
            if delete_folder and folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

    # ---------------------------------------------------------------- #
    # Setup helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def _cleanup_simulation_folders() -> None:
        """Remove any ``En_*`` directories left in the current directory."""
        for item in os.listdir("."):
            if item.startswith("En_") and os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)

    def _stage_input(self, input: dict | str, folder: Path, idn: int) -> str:
        """
        Render a Mako template or copy a static ``.DATA`` file into ``folder``.

        Parameters
        ----------
        input : dict or str
            Either Mako parameters or a path to a ``.DATA`` deck.
        folder : Path
            Destination directory.
        idn : int
            Ensemble index (injected into the Mako context as ``member``).

        Returns
        -------
        str
            Basename of the deck file inside ``folder``.

        Raises
        ------
        FileNotFoundError
            If a string input does not point to an existing file.
        ValueError
            If a string input is not a ``.DATA`` file.
        TypeError
            For unsupported input types.
        """
        if isinstance(input, dict):
            input["member"] = idn
            return self.render_makofile(self.makofile, str(folder), input)

        if isinstance(input, str):
            if not os.path.isfile(input):
                raise FileNotFoundError(f"Input file {input} not found")
            if not input.endswith(".DATA"):
                raise ValueError("Input string must be a path to a .DATA file")
            datafile = os.path.basename(input)
            shutil.copy(input, folder)
            return datafile

        raise TypeError(f"Unsupported input type: {type(input).__name__}")

    @staticmethod
    def _setup_case(datafile: str, julia):
        """Invoke JutulDarcy's case-setup routine with output suppressed."""
        return _suppress_julia(julia, f'setup_case_from_data_file("{datafile}")')

    @staticmethod
    def _detect_units(case, julia) -> str | Any:
        """
        Detect the unit system from the deck's RUNSPEC section.

        Returns
        -------
        str
            One of ``"metric"``, ``"si"``, ``"field"`` if specified.
        Any
            ``julia.missing`` if no unit keyword is present.
        """
        runspec = case.input_data["RUNSPEC"]
        for key, name in (("METRIC", "metric"), ("SI", "si"), ("FIELD", "field")):
            if julia.haskey(runspec, key):
                return name
        return julia.missing

    @staticmethod
    def _extract_actnum(case, julia=None) -> np.ndarray:
        """
        Return the flat (1/0) mask of cells the simulation actually carries.

        This follows the *processed* mesh rather than the deck's ACTNUM. Mesh
        processing (pinch-out collapse, geometry repair) can drop cells that
        ACTNUM marks active -- the coarsened Drogon grid loses one at
        (19, 6, 12) -- and gradients come back on the processed mesh. Masking
        with the deck's ACTNUM then either raises ("Parameter length does not
        match number of active cells") or, if the counts happened to agree,
        silently shifts every gradient value past the dropped cell onto its
        neighbour.

        Falls back to the deck's ACTNUM when the mesh exposes no cell map,
        and to all-active when the deck has no ACTNUM either.

        Parameters
        ----------
        case : Any
            Julia case object.
        julia : juliacall.Main, optional
            Julia main module. Without it the deck's ACTNUM is used, which is
            only correct when mesh processing dropped nothing.

        Returns
        -------
        np.ndarray
            Flat (Fortran-ordered) 1/0 vector over the full Cartesian grid.
        """
        nx, ny, nz = case.input_data["GRID"]["cartDims"]
        ncell = int(nx) * int(ny) * int(nz)

        if julia is not None:
            try:
                julia.case = case
                cell_map = np.asarray(julia.seval(
                    "let m = Jutul.physical_representation("
                    "JutulDarcy.reservoir_domain(case.model)); "
                    "[Int(c) for c in m.cell_map] end"
                ), dtype=np.int64) - 1
            except Exception:
                cell_map = None
            if cell_map is not None and cell_map.size:
                # `_active_to_full_grid` fills the mask in ascending index
                # order, so a cell map that is not sorted would scramble the
                # gradient. Say so rather than return wrong numbers.
                if not np.all(np.diff(cell_map) > 0):
                    raise ValueError(
                        "Reservoir cell map is not strictly ascending; the "
                        "active-cell mask cannot represent it. The gradient "
                        "would be scattered onto the wrong cells."
                    )
                actnum = np.zeros(ncell, dtype=np.int64)
                actnum[cell_map] = 1
                return actnum

        try:
            actnum = np.array(case.input_data["GRID"]["ACTNUM"])
            return actnum.flatten(order="F")
        except (KeyError, AttributeError):
            # No ACTNUM keyword in the deck => every cell is active.
            return np.ones(ncell)

    def _format_output(self, pyres: pd.DataFrame):
        """Convert the per-member DataFrame to the user-requested container."""
        if self.output_format == "dataframe":
            return pyres
        if self.output_format == "dict":
            return pyres.to_dict(orient="list")
        return pyres.to_dict(orient="records")  # 'list' format → list of records

    @staticmethod
    def render_makofile(makofile: str, folder: str, input: dict) -> str:
        """
        Render a Mako template into a ``.DATA`` file in ``folder``.

        Parameters
        ----------
        makofile : str
            Path to the ``.mako`` template.
        folder : str
            Destination directory.
        input : dict
            Template context variables.

        Returns
        -------
        str
            Basename of the rendered ``.DATA`` file.
        """
        datafile = os.path.basename(makofile).replace(".mako", ".DATA")
        template = Template(filename=makofile)
        with open(os.path.join(folder, datafile), "w") as f:
            f.write(template.render(**input))
        return datafile

    # ---------------------------------------------------------------- #
    # Datatype extraction
    # ---------------------------------------------------------------- #
    def extract_datatypes(self, jlres, jlcase, units, julia) -> pd.DataFrame:
        """
        Extract requested summary keywords from a finished simulation.

        Parameters
        ----------
        jlres : Any
            Julia result object from ``simulate_reservoir``.
        jlcase : Any
            Julia case object (used for the start date).
        units : str or Any
            Unit-system identifier (see :meth:`_detect_units`).
        julia : juliacall.Main
            Julia main module.

        Returns
        -------
        pd.DataFrame
            One row per report point; one column per datatype. ``df.attrs``
            holds per-column unit strings.

        Raises
        ------
        ValueError
            If a requested datatype or well is missing from the results.
        """
        jl_units = julia.Symbol(units) if isinstance(units, str) else units
        smry = julia.JutulDarcy.summary_result(jlcase, jlres, jl_units)

        # JutulDarcy builds its time vector as a floating-point cumulative sum
        # over adaptive ministeps (`output_substates=True` keeps every one of
        # them), so a report boundary can land a fraction of a second below the
        # whole second we asked for. Round to the nearest second -- truncating
        # would turn e.g. 10367999.9999 into day 119, silently dropping the
        # day-120 report point.
        sim_seconds = np.rint(
            np.array(list(smry["TIME"].seconds), dtype=np.float64)
        ).astype(np.int64)
        self.start_date = jlcase.input_data["RUNSPEC"]["START"]
        self.report_seconds = self._compute_report_seconds()

        # Look each requested report point up explicitly rather than
        # intersecting: this keeps the rows in the order given by
        # `self.index[1]` and yields exactly one row per report point.
        step_lookup = {t: i for i, t in enumerate(sim_seconds.tolist())}
        missing = [t for t in self.report_seconds.tolist() if t not in step_lookup]
        if missing:
            last_sim = int(sim_seconds[-1])
            last_req = int(self.report_seconds[-1])
            if last_sim < last_req:
                cause = (
                    f"The run stopped at day {last_sim / SECONDS_PER_DAY:g} "
                    f"of {last_req // SECONDS_PER_DAY}, so it aborted early. "
                    f"Jutul returns partial results silently unless "
                    f"`error_on_incomplete` is set."
                )
            else:
                cause = (
                    f"The run did reach the final report point (day "
                    f"{last_req // SECONDS_PER_DAY}), so these times are not "
                    f"report steps in the deck's SCHEDULE. Every requested "
                    f"report point needs a matching DATES/TSTEP entry -- "
                    f"the simulator only writes summary output at the report "
                    f"steps the deck defines."
                )
            raise ValueError(
                f"Simulation produced "
                f"{len(self.report_seconds) - len(missing)} of "
                f"{len(self.report_seconds)} requested report points. "
                f"Missing at day(s) "
                f"{[t // SECONDS_PER_DAY for t in missing]}. {cause}"
            )
        idx = np.array([step_lookup[t] for t in self.report_seconds.tolist()])

        res: dict[str, np.ndarray] = {}
        attrs: dict[str, str] = {}

        wells = smry["VALUES"]["WELLS"]
        field = smry["VALUES"]["FIELD"]

        for datatype in self.datatype:
            if ":" in datatype:
                # Well-level datatype: "<baseID>:<wellID>"
                baseID, wellID = datatype.split(":")
                if wellID not in wells:
                    raise ValueError(
                        f"Well ID '{wellID}' not found for datatype '{baseID}'"
                    )
                well_data = wells[wellID]
                if baseID not in well_data:
                    raise ValueError(
                        f"Datatype '{baseID}' not found for well '{wellID}'"
                    )
                res[datatype] = np.array(well_data[baseID])[idx]
                attrs[datatype] = get_metric_unit(baseID)
            else:
                # Field-level datatype.
                if datatype not in field:
                    raise ValueError(
                        f"Datatype '{datatype}' not found in field results"
                    )
                res[datatype] = np.array(field[datatype])[idx]
                attrs[datatype] = get_metric_unit(datatype)

        df = pd.DataFrame(res, index=self.index[1])
        df.index.name = self.index[0]
        df.attrs = attrs
        return df

    def _compute_report_seconds(self) -> np.ndarray:
        """
        Convert the configured report points to integer seconds.

        Returns
        -------
        np.ndarray
            Seconds elapsed from the simulation start for each report point.

        Raises
        ------
        ValueError
            If ``self.report_type`` is not recognised (defensive; should be
            blocked by :meth:`__init__` validation).
        """
        rtype, rpoints = self.index
        if rtype == "days":
            return np.array(rpoints, dtype=np.int64) * SECONDS_PER_DAY
        if rtype == "dates":
            return np.array(
                [(d - self.start_date).total_seconds() for d in rpoints],
                dtype=np.int64,
            )
        raise ValueError(f"Invalid report type: {rtype}")

    # ---------------------------------------------------------------- #
    # Adjoint computation
    # ---------------------------------------------------------------- #
    def _compute_adjoints(self, case, jlres, pyres, units, actnum_vec,
                          idn: int, julia) -> pd.DataFrame:
        """
        Compute gradients for all configured adjoint objectives.

        Parameters
        ----------
        case : Any
            Julia case object.
        jlres : Any
            Julia simulation result.
        pyres : pd.DataFrame
            Forward-extracted results, used as a sanity check against
            Jutul's own objective evaluation.
        units : Any
            Unit system identifier (stored in returned ``df.attrs``).
        actnum_vec : np.ndarray
            Flat ACTNUM vector for full-grid embedding.
        idn : int
            Ensemble index (used only for pbar positioning).
        julia : juliacall.Main
            Julia main module.

        Returns
        -------
        pd.DataFrame
            Index: requested adjoint evaluation points;
            columns: ``(objective, parameter)`` MultiIndex.
        """
        # Optimization mode requires setting up a parameter dictionary and
        # freeing the parameters Jutul will differentiate w.r.t.
        if self.adjoint_mode == "optimization":
            julia.grad_case = julia.seval(
                "JutulDarcy.setup_reservoir_dict_optimization(case, verbose=false)"
            )
            julia.seval("JutulDarcy.free_optimization_parameters!(grad_case)")

        # Pre-allocate output containers.
        grad_dict: dict[tuple[str, str], list] = {
            (col, p): []
            for col, obj in self.adjoint_info.items()
            for p in obj.parameters
        }
        func_dict: dict[str, list] = (
            {col: [] for col in self.adjoint_info} if self.eval_adjoint_funcs else {}
        )

        # dWI/dK depends only on the case, so it is built once and contracted
        # with each objective's dJ/dWI below.
        needs_perm = any(
            "perm" in param.lower()
            for obj in self.adjoint_info.values()
            for param in obj.parameters
        )
        if needs_perm:
            wi_jacobian, ncells = _well_index_perm_jacobian(
                case, julia, self.well_index_from_perm
            )
            perm_copy = _detect_copied_perm_axes(
                case, actnum_vec, self.perm_copied, julia
            )
        else:
            wi_jacobian, ncells, perm_copy = {}, 0, None

        # Build the adjoint storage and the ministep expansion once per member.
        # `solve_adjoint_sensitivities` does both on every call, which is pure
        # repetition when a member has many objectives. `step_times` then lets
        # each objective stop its backward sweep at its own evaluation point.
        step_times = None
        if self.adjoint_mode == "sensitivities" and self.adjoint_reuse_storage:
            julia.adj_packed = _suppress_julia(
                julia, "subsurface_adjoint_packed(case, res)"
            )
            julia.adj_storage = _suppress_julia(
                julia, "subsurface_adjoint_storage(case)"
            )
            step_times = np.asarray(
                _suppress_julia(julia, "subsurface_packed_step_times(adj_packed)"),
                dtype=float,
            )

        pbar = self._make_adjoint_pbar(idn)
        sim_times = np.array(jlres.time) # Unit: seconds
        adjoint_index_final = None  # captured from the last objective

        for col, info in pbar:
            # Resolve the evaluation points for this objective.
            adj_seconds, adj_index = self._resolve_adjoint_steps(info)
            adjoint_index_final = adj_index

            # Closest simulation step to each requested time (Julia is 1-indexed).
            adj_step_idx = [
                int(np.argmin(np.abs(sim_times - s))) + 1 for s in adj_seconds
            ]

            funcs = well_QOI_objective(
                info.wellID, info.phase, adj_seconds,
                adj_step_idx, info.is_rate, julia=julia,
            )

            if self.adjoint_pbar:
                pbar.set_description_str(f"Adjoints for {col}")

            for i, func in enumerate(funcs):
                julia.func = func

                # The objective is zero after its own evaluation point, so the
                # adjoint is zero there too: sweep no further back than that.
                n_steps = None
                if step_times is not None:
                    n_steps = int(
                        np.searchsorted(step_times, adj_seconds[i], side="right")
                    ) or len(step_times)

                grad, well_sens = self._solve_adjoint(julia, n_steps=n_steps)

                # Well indices derived from the permeability field contribute
                # (dJ/dWI)·(dWI/dK), which JutulDarcy's permeability gradient
                # leaves out.
                wi_chain_term = (
                    _well_index_chain_term(wi_jacobian, well_sens, ncells, julia)
                    if wi_jacobian else None
                )

                # Cross-check Julia's objective evaluation against the value
                # we already obtained from the forward extraction.
                func_val = julia.Jutul.evaluate_objective(func, case, jlres.result)
                expected = pyres.loc[adj_index[i]][col]
                assert np.isclose(func_val, expected), (
                    f"func_val: {func_val:.3e}, pyres: {expected:.3e}"
                )
                if self.eval_adjoint_funcs:
                    func_dict[col].append(func_val)

                # Post-process the raw Julia gradient for each requested parameter.
                for param in info.parameters:
                    grad_dict[(col, param)].append(
                        _extract_adjoint(grad, case, param, actnum_vec,
                                         perm_copy, julia,
                                         wi_chain_term=wi_chain_term)
                    )

        if self.adjoint_pbar:
            pbar.close()

        # Wrap the gradients into a multi-indexed DataFrame.
        cols = pd.MultiIndex.from_tuples(grad_dict.keys())
        adjoints = pd.DataFrame(grad_dict, columns=cols, index=adjoint_index_final)
        adjoints.index.name = self.index[0]
        adjoints.attrs = {"units": units}

        if self.eval_adjoint_funcs:
            fun = pd.DataFrame(func_dict, index=adjoints.index)
            fun.index.name = adjoints.index.name
            self.adjoint_funcs = fun

        return adjoints

    def _solve_adjoint(self, julia, n_steps: int | None = None):
        """
        Invoke the appropriate JutulDarcy adjoint solver for the current mode.

        Parameters
        ----------
        julia : juliacall.Main
            Julia main module. The names ``case``, ``res``, ``grad_case`` and
            ``func`` are expected to be already bound in the Julia namespace.
        n_steps : int, optional
            Stop the backward sweep after this many substeps, reusing the
            pre-built ``adj_storage`` / ``adj_packed`` bindings. When omitted,
            each call builds its own storage and sweeps the whole horizon.

        Returns
        -------
        tuple
            ``(gradient, well_sensitivities)``. The first element is the raw
            Julia gradient object searched by :func:`_extract_adjoint`; the
            second is a Julia mapping from well name to that well's parameter
            sensitivities (used for the well-index chain rule), or ``None`` if
            the mode does not expose them.
        """
        if self.adjoint_mode == "sensitivities":
            # `reservoir_sensitivities` drops everything outside the reservoir
            # submodel, so use our own wrapper that keeps the well
            # sensitivities from the same adjoint solve.
            if n_steps is None:
                out = _suppress_julia(
                    julia, "subsurface_sensitivities_with_wells(case, res, func)"
                )
            else:
                out = _suppress_julia(
                    julia,
                    "subsurface_sensitivities_reuse("
                    f"case, adj_storage, adj_packed, func, {n_steps})"
                )
            return out.reservoir, out.models

        grad = _suppress_julia(
            julia,
            "JutulDarcy.parameters_gradient_reservoir(grad_case, func, deps=:case)"
        )
        # The optimization dict mirrors `setup_reservoir_dict_optimization`, so
        # the per-well entries already sit under a `:wells` key.
        return grad, _extract_key_value(grad, "wells", julia)

    def _resolve_adjoint_steps(self, info: AdjointObjective):
        """
        Convert an objective's step specification into seconds + index labels.

        Parameters
        ----------
        info : AdjointObjective
            Spec containing ``steps`` (``"all"``, list of ints, or list of
            :class:`datetime.datetime`).

        Returns
        -------
        tuple
            ``(seconds_array, index_labels)`` ready for adjoint evaluation.

        Raises
        ------
        TypeError
            If the step list contains an unsupported element type.
        """
        if info.steps == "all":
            return self.report_seconds, self.index[1]

        first = info.steps[0]
        if isinstance(first, int):
            return (
                np.array(info.steps, dtype=np.int64) * SECONDS_PER_DAY,
                info.steps,
            )
        if isinstance(first, dt.datetime):
            return (
                np.array(
                    [(d - self.start_date).total_seconds() for d in info.steps],
                    dtype=np.int64,
                ),
                info.steps,
            )
        raise TypeError(f"Unsupported adjoint step type: {type(first).__name__}")

    def _make_adjoint_pbar(self, idn: int):
        """
        Build an iterator over ``self.adjoint_info``, optionally with a pbar.

        Parameters
        ----------
        idn : int
            Ensemble index (used to stagger pbar positions in parallel runs).

        Returns
        -------
        Iterable
            Either ``self.adjoint_info.items()`` or a wrapped :class:`tqdm`.
        """
        if not self.adjoint_pbar:
            return self.adjoint_info.items()

        # Drop the default colour so we can override it for the adjoint bars.
        opts = {k: v for k, v in PBAR_OPTS.items() if k != "colour"}
        return tqdm(
            self.adjoint_info.items(),
            desc="Solving adjoints",
            position=idn + 1,
            leave=False,
            unit="obj",
            dynamic_ncols=False,
            colour="#713996",
            **opts,
        )