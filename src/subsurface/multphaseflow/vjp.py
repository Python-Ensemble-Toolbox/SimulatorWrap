"""PET's labelled seeds for scalar simulator summaries; columns are independent objectives."""

import numpy as np


def scalar_seeds(weights, reports, datatypes):
    if not weights.index.is_unique or not weights.columns.is_unique:
        raise ValueError("VJP report and datatype labels must be unique")
    cells, vectors = [], []
    for label, row in weights.iterrows():
        for key, value in row.items():
            if value is None:
                continue
            if label not in reports or key not in datatypes:
                raise ValueError(f"Unknown VJP observation {(label, key)}")
            vector = np.atleast_1d(np.asarray(value, dtype=float))
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError(f"Expected finite scalar-observation seeds at {(label, key)}")
            cells.append((label, key))
            vectors.append(vector)
    if not cells:
        raise ValueError("VJP needs at least one observed cell")
    return cells, np.stack(vectors)


def weighted_sums(prediction, reports, cells, seeds):
    """The weighted sums of ``scalar_seeds`` cells in a wrapper's forward output (a frame, or records per report)."""
    values = []
    for label, key in cells:
        if hasattr(prediction, "at"):
            values.append(prediction.at[label, key])
        else:
            values.append(prediction[list(reports).index(label)][key])
    return np.array([float(np.ravel(v)[0]) for v in values]) @ seeds


def finite_difference_check(forward, gradients, state, sums, step=1e-2, rtol=0.05, seed=0):
    """Compare ``gradients`` with central differences of the weighted sums along random directions.

    ``forward(state)`` returns the sums, ``(k,)``; ``gradients[name]`` is ``(k, n)``. Each state variable is
    perturbed along a +-1 direction (every entry moves, also those next to wells) and along a uniform shift,
    additively for ``log_`` variables and relative to its value otherwise. Two step sizes separate the error of
    the difference quotient from a derivative that is missing a term. Returns one row per variable, direction
    and sum; ``ok`` is ``False`` for a mismatch and ``None`` where the two step sizes disagree too much to tell.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, x in state.items():
        x = np.asarray(x, dtype=float)
        scale = np.ones_like(x) if name.startswith("log_") else x
        for kind, pattern in (("alternating", rng.choice([-1.0, 1.0], size=x.shape)), ("uniform", np.ones_like(x))):
            direction = pattern * scale
            quotients = []
            for h in (step, step / 2):
                plus = forward({**state, name: x + h * direction})
                minus = forward({**state, name: x - h * direction})
                quotients.append((np.asarray(plus) - np.asarray(minus)) / (2 * h))
            adjoint = np.asarray(gradients[name]) @ direction
            for j, (coarse, fine, value) in enumerate(zip(*quotients, adjoint)):
                spread = abs(coarse - fine)
                tolerance = rtol * max(abs(fine), abs(value))
                ok = bool(abs(value - fine) <= tolerance + 2 * spread)
                if spread > tolerance:
                    ok = None if ok else False
                rows.append({"variable": name, "direction": kind, "sum": j, "adjoint": float(value),
                             "finite_difference": float(fine), "step_spread": float(spread),
                             "base": float(sums[j]), "ok": ok})
    return rows
