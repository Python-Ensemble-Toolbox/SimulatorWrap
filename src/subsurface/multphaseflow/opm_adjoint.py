"""Serial PET wrapper for OPM Flow with opm-adjoint: forecasts, and weighted adjoints through its JSON objectives."""

from contextlib import closing
import json
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import pandas as pd
from resdata.summary import Summary
from resdata.resfile import ResdataFile

from .opm import flow
from .vjp import finite_difference_check, keywords_changed, scalar_seeds, weighted_sums

PHASES = {"WOPR": "oil", "WWPR": "water", "WGPR": "gas"}
FIELDS = {"poro", "permx", "permy", "permz", "log_permx", "log_permy", "log_permz"}
# INIT arrays Flow computes from a field; opm-adjoint carries their derivatives itself.
DERIVED = {"PORO": {"PORV"}, "PERMX": {"TRANX", "TRANY", "TRANZ"}, "PERMY": {"TRANX", "TRANY", "TRANZ"},
           "PERMZ": {"TRANX", "TRANY", "TRANZ"}}
HEADERS = {"INTEHEAD", "LOGIHEAD", "DOUBHEAD"}
# Flags the wrapper sets itself; the adjoint's solver settings (--adjoint-linear-solver...) remain the user's.
OWNED_FLAGS = ("--adjoint-file", "--adjoint-save", "--adjoint-mode", "--adjoint-config", "--adjoint-objective",
               "--output-dir", "--enable-storage-cache")


class FlowAdjoint(flow):
    """Use the same Flow executable, deck renderer and flags for predictions and VJPs."""

    def __init__(self, input_file):
        super().__init__(input_file)
        self.executable = input_file.get("adjoint_executable", "flow_adjoint")
        if self.options["mpi"] or self.options["mpiarray"] or input_file.get("hpc") or self.upscale:
            raise ValueError("FlowAdjoint supports serial, non-upscaled decks only")
        self.flags = shlex.split(self.options["sim_flag"])
        if any(f.startswith(OWNED_FLAGS) for f in self.flags):
            raise ValueError("FlowAdjoint owns the archive, mode, objective, output and storage-cache flags")
        if self.options["parsing-strictness"]:
            self.flags.append("--parsing-strictness=" + self.options["parsing-strictness"])
        self.chain_rule = None  # the report of check_chain_rule, once it has run

    def _flow(self, deck, output, *flags):
        output.mkdir(exist_ok=True)
        with (output / "flow.log").open("w+") as log:
            result = subprocess.run([self.executable, str(deck), f"--output-dir={output}",
                                     *self.flags, "--enable-storage-cache=false", *flags],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=self.options.get("sim_limit"))
            if result.returncode:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 2000))
                raise RuntimeError(f"flow_adjoint exited {result.returncode}: {log.read()}")

    def call_sim(self, folder, wait_for_proc=False):
        folder = Path(folder).resolve()
        try:
            self._flow(folder / (self.file + ".DATA"), folder)
        except (RuntimeError, subprocess.TimeoutExpired):
            return False
        return True

    def run_vjp(self, state, weights):
        """Gradients of weighted sums of producer rates: ``{name: (k, n)}`` for the ``k`` weight columns."""
        cells, seeds = scalar_seeds(weights, self.true_order[1], self.all_data_types)
        if not state or set(state) - FIELDS:
            raise ValueError(f"Expected direct cell fields from {sorted(FIELDS)}, got {list(state)}")
        if len({key.removeprefix("log_") for key in state}) != len(state):
            raise ValueError("Only one PET variable may control a deck field")
        for _, key in cells:
            kind, _, well = key.partition(":")
            if kind.upper() not in PHASES or not re.fullmatch(r"[\w.-]+", well):
                raise ValueError(f"Unsupported production rate: {key}")
        if self.chain_rule is None or set(self.chain_rule["ties"]) != set(state):
            self.check_chain_rule(state)
        gradients = {key: np.zeros((seeds.shape[1], np.size(x))) for key, x in state.items()}
        with TemporaryDirectory(prefix="vjp_", dir=".") as tmp:
            root = Path(tmp).resolve()
            self._runMako(str(root) + "/", dict(state, member=0, level=self.level))
            deck = root / (self.file + ".DATA")
            # The substep lengths come from the summary, which RPTONLY would thin to the report steps
            deck.write_text(re.sub(r"(?m)^[ \t]*RPTONLY[ \t]*$", "-- RPTONLY", deck.read_text()))
            archive = f"--adjoint-file={root / 'archive'}"
            self._flow(deck, root, archive, "--adjoint-save=true")
            summary = Summary(str(root / self.file))
            days = np.asarray(summary.days, dtype=float)
            days = days[days > 0]
            if len(days) != len(list((root / "archive" / "adjoint" / "substep").iterdir())):
                raise ValueError("Expected one summary row per recorded substep; check summary thinning")
            key_map = {k.upper(): k for k in summary.keys()}
            start = pd.Timestamp(summary.start_time)
            terms, rates, steps = [], [], []
            for label, key in cells:
                key = key_map[key.upper()]
                if summary.unit(key).upper() != "SM3/DAY":
                    raise ValueError(f"Expected metric surface rate for {key}, got {summary.unit(key)}")
                time = (float(label) if self.true_order[0] == "days"
                        else (pd.Timestamp(label) - start) / pd.Timedelta(days=1))
                i = int(np.argmin(abs(days - time)))
                step = days[i] - (days[i - 1] if i else 0.0)
                after = days[i + 1] - days[i] if i + 1 < len(days) else step
                # Summary times are single precision: the window must be resolved by them.
                if abs(days[i] - time) > 1e-3 or min(step, after) < 16 * np.spacing(np.float32(days[i])):
                    raise ValueError(f"No simulator step ends on the report at {label}")
                kind, well = key.split(":")
                # A term is active on the substep whose end lies in its window.
                terms.append({"kind": "rate", "wells": [well], "phase": PHASES[kind],
                              "window_days": [days[i] - step / 2, days[i] + after / 2]})
                rates.append(summary.numpy_vector(key)[len(summary.days) - len(days) + i])
                steps.append((i, step * 86400.0))
            rates = np.asarray(rates, dtype=float)

            # A rate term is weight * dt * q, with q the internal rate (m3/s, negative for a
            # producer). Its unit-weight value gives dt of each report's substep in double
            # precision; the summary's time differences only check it.
            values = self._objective_terms(deck, root, archive, terms)
            seconds = {}
            for (i, step), value, rate in zip(steps, values, rates):
                if rate:
                    seconds.setdefault(i, []).append(-86400.0 * value / rate)
            resolution = 86400.0 * 2 * np.spacing(np.float32(days[-1]))
            for i, estimates in seconds.items():
                step = next(step for j, step in steps if j == i)
                if not np.allclose(estimates, step, rtol=1e-3, atol=resolution):
                    raise ValueError("Rate objectives do not match the forward summary (units, sign or timing)")
                seconds[i] = float(np.median(estimates))
            unit = np.array([-86400.0 / seconds.get(i, step) for i, step in steps])

            with closing(ResdataFile(str(root / (self.file + ".INIT")))) as init:
                cells_total = int(np.prod(init["INTEHEAD"][0][8:11]))
                deck_fields = {key: np.asarray(init[key.removeprefix("log_").upper()][0]) for key in state}

            for j, column in enumerate(seeds.T):
                if not column.any():
                    continue
                config = root / "objective.json"
                config.write_text(json.dumps({"schema_version": 1, "objective": {"terms": [
                    dict(term, weight=w * u) for term, w, u in zip(terms, column, unit) if w]}}))
                with TemporaryDirectory(dir=root) as replay:
                    out = Path(replay)
                    self._flow(deck, out, archive, "--adjoint-mode=gradient", f"--adjoint-config={config}")
                    prefix = str(out / self.file) + ".ADJOINT_GRADIENTS"
                    actual = json.loads(Path(prefix + ".json").read_text())["objective"]["value"]
                    # UNSMRY is single precision: scale roundoff by the terms before cancellation.
                    if not np.isclose(actual, column @ rates, rtol=2e-5,
                                      atol=1e-8 + 1e-6 * (abs(column) @ abs(rates))):
                        raise ValueError("Replayed objective does not match the weighted forward summary")
                    poro = np.loadtxt(prefix + "_PORO.csv", delimiter=",", skiprows=1, ndmin=2)
                    perm = np.loadtxt(prefix + "_PERM.csv", delimiter=",", skiprows=1, ndmin=2)
                order = np.argsort(poro[:, 0])
                active = poro[order, 0].astype(int)
                if not np.array_equal(active, np.sort(perm[:, 0]).astype(int)):
                    raise ValueError("Porosity and permeability gradients cover different cells")
                poro, perm = poro[order, 1], perm[np.argsort(perm[:, 0]), 1:] * 9.869233e-16
                for name, x in state.items():
                    x = np.asarray(x, dtype=float)
                    if x.size not in (len(active), cells_total):
                        raise ValueError(f"{name} must have one value per active cell or per grid cell")
                    at = active if x.size == cells_total else slice(None)
                    expected = np.exp(x[at]) if name.startswith("log_") else x[at]
                    if (expected.shape != deck_fields[name].shape or not np.all(expected > 0)
                            or not np.allclose(deck_fields[name], expected, rtol=2e-5, atol=0)):
                        raise ValueError(f"{name} must render directly in grid order (check units and clipping)")
                    if name == "poro":
                        g = poro / expected
                    else:  # the deck may tie other axes to this one: PERMb = factor * PERMa
                        g = perm[:, "xyz".index(name[-1])] + sum(
                            factor * perm[:, "xyz".index(axis)] for axis, factor in self.chain_rule["ties"][name].items())
                    if name.startswith("log_"):
                        g = g * expected
                    if not np.isfinite(g).all():
                        raise ValueError(f"Nonfinite replay gradient for {name}")
                    gradients[name][j, at] = g
        return gradients

    def _objective_terms(self, deck, root, archive, terms):
        """Values of unit-weight objective ``terms`` on the recorded trajectory."""
        config = root / "terms.json"
        config.write_text(json.dumps({"schema_version": 1, "objective": {"terms": terms}}))
        with TemporaryDirectory(dir=root) as replay:
            self._flow(deck, Path(replay), archive, "--adjoint-mode=objective", f"--adjoint-config={config}")
            log = (Path(replay) / "flow.log").read_text()
        values = re.findall(r"Adjoint objective term\s+\d+\s+\S+\s+J_term = (\S+)", log)
        if len(terms) == 1:  # the tool lists the terms only when there are several
            values = re.findall(r"Adjoint objective J = (\S+)", log)
        if len(values) != len(terms):
            raise RuntimeError("Could not read the objective terms from the flow_adjoint log")
        return np.array(values, dtype=float)

    # ------------------------------------------------------------------
    # Does the derivative resolve every term of the chain rule?
    # ------------------------------------------------------------------
    def check_chain_rule(self, state, weights=None, step=1e-2, rtol=0.05, seed=0):
        """Find everything in the deck that moves with ``state``, and say whether ``run_vjp`` differentiates it.

        Each state variable is perturbed, the deck rendered again and read by Flow without simulating. The
        rendered text may differ only inside the keyword of the variable's own field, and the grid arrays Flow
        reports only in that field, in what Flow derives from it, and in permeability axes the deck ties to it
        by a constant factor (``COPY``/``MULTIPLY``), which ``run_vjp`` then adds. Anything else is a term the
        derivative lacks, and raises. Connection factors left for Flow to compute from the permeability are
        listed as ``approximate``: opm-adjoint keeps their leading term.

        With ``weights`` (as for ``run_vjp``) the gradients are also compared with central differences of the
        weighted sums through template and simulator: the check of the whole chain, at four simulations per
        variable and direction. A mismatch raises; the rows are in the returned report.
        """
        if not state or set(state) - FIELDS:
            raise ValueError(f"Expected direct cell fields from {sorted(FIELDS)}, got {list(state)}")
        rng = np.random.default_rng(seed)
        report = {"ties": {}, "derived": {}, "approximate": [], "unresolved": [], "finite_difference": []}
        base_text, base, _, wells = self._dry_run(state)
        if wells and any(name != "poro" for name in state):
            report["approximate"].append(
                f"connection factors of {wells} follow the permeability; opm-adjoint differentiates the "
                "leading (Kh) term only")
        for name, x in state.items():
            x = np.asarray(x, dtype=float)
            target = name.removeprefix("log_").upper()
            sign = rng.choice([-1.0, 1.0], size=x.shape)
            moved = x + 1e-3 * sign if name.startswith("log_") else x * (1 + 1e-3 * sign)
            text, arrays, cells, _ = self._dry_run({**state, name: moved})
            touched = keywords_changed(base_text, text)
            if touched is None:
                report["unresolved"].append(f"{name} changes the layout of the rendered deck")
            else:
                for keyword in sorted(touched - {target}):
                    report["unresolved"].append(f"the template writes {keyword} from {name}")
            changed = {key for key in base if key not in arrays or not np.array_equal(arrays[key], base[key])}
            if target not in changed:
                report["unresolved"].append(f"{name} does not reach {target} in the deck Flow reads")
                continue
            active = arrays[target] != base[target]
            expected = np.exp(moved) if name.startswith("log_") else moved
            if expected.size != base[target].size:  # one value per grid cell
                expected = expected[cells]
            if not np.allclose(arrays[target], expected, rtol=2e-5, atol=0):
                report["unresolved"].append(
                    f"{target} is not {name} itself in {int(np.sum(~np.isclose(arrays[target], expected, rtol=2e-5, atol=0)))} "
                    "cells (clipping or another transform)")
            ties = {}
            for axis in "xyz":
                other = "PERM" + axis.upper()
                if target.startswith("PERM") and other != target and other in changed:
                    factor = float(np.median(base[other][active] / base[target][active]))
                    tied = all(np.allclose(a[other], factor * a[target], rtol=1e-5, atol=0) for a in (base, arrays))
                    if tied and not {other.lower(), "log_" + other.lower()} & set(state):
                        ties[axis] = factor
                    else:
                        report["unresolved"].append(f"{other} follows {name}, but not by a constant factor of its own")
            report["ties"][name] = ties
            allowed = {target} | DERIVED[target] | {"PERM" + axis.upper() for axis in ties}
            report["derived"][name] = sorted(changed & (allowed - {target}))
            for key in sorted(changed - allowed):
                report["unresolved"].append(f"{key} follows {name}")
        if report["unresolved"]:
            raise ValueError("run_vjp would miss part of the chain rule: " + "; ".join(report["unresolved"]))
        for note in report["approximate"]:
            warnings.warn(note, stacklevel=2)
        self.chain_rule = report
        if weights is not None:
            cells, seeds = scalar_seeds(weights, self.true_order[1], self.all_data_types)
            member = iter(range(10**6))

            def forward(x):
                records = self.run_fwd_sim({key: np.array(value, dtype=float) for key, value in x.items()}, next(member))
                return weighted_sums(records, self.true_order[1], cells, seeds)

            gradients = self.run_vjp({key: np.array(value, dtype=float) for key, value in state.items()}, weights)
            report["finite_difference"] = finite_difference_check(forward, gradients, state, forward(state), step, rtol, seed)
            failed = [row for row in report["finite_difference"] if row["ok"] is False]
            if failed:
                raise ValueError(f"Gradients differ from finite differences beyond {rtol:g}: {failed[:4]}")
            if any(row["ok"] is None for row in report["finite_difference"]):
                warnings.warn("The finite differences change too much with the step to judge some gradients: the "
                              "simulated rates are not smooth in the state. Tighten the solver tolerances or fix "
                              "the time steps (rows with ok=None in the report).", stacklevel=2)
        return report

    def _dry_run(self, state):
        """Render ``state`` and let Flow read the deck without simulating.

        Returns the deck text, its INIT arrays, the active cells, and the wells whose connection factors Flow
        computes from the permeability. The deck sits where a member's deck does, so relative includes resolve.
        """
        with TemporaryDirectory(prefix="audit_", dir=".") as tmp:
            folder = Path(tmp).resolve()
            self._runMako(str(folder) + "/", dict({key: np.array(value, dtype=float) for key, value in state.items()},
                                                  member=0, level=self.level))
            deck = folder / (self.file + ".DATA")
            self._flow(deck, folder, "--enable-dry-run=true")
            with closing(ResdataFile(str(folder / (self.file + ".INIT")))) as init:
                arrays = {key: np.asarray(init[key][0]).copy() for key in dict.fromkeys(init.keys()) if key not in HEADERS}
            with closing(ResdataFile(str(folder / (self.file + ".EGRID")))) as grid:
                cells = np.asarray(grid["ACTNUM"][0]) != 0 if grid.has_kw("ACTNUM") else slice(None)
            return deck.read_text(), arrays, cells, _connection_factors_from_permeability(deck)


def _connection_factors_from_permeability(deck):
    """Wells with a COMPDAT connection whose transmissibility factor (item 8) and Kh (item 10) are both defaulted."""
    wells, seen = set(), set()

    def scan(path):
        if path in seen or not path.is_file():
            return
        seen.add(path)
        keyword = ""
        text = re.sub(r"--.*", "", path.read_text(errors="replace"))
        for include in re.findall(r"(?m)^\s*INCLUDE\s*\n\s*'?([^'/\s][^'\n]*?)'?\s*/", text):
            scan((path.parent / include.strip()).resolve())
        for line in text.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"[A-Z][A-Z0-9_]{1,7}", stripped):
                keyword = stripped
            elif keyword in ("COMPDAT", "COMPDATL") and stripped and stripped != "/":
                items = []
                for token in stripped.split("/")[0].split():
                    repeat = re.fullmatch(r"(\d+)\*", token)
                    items += ["*"] * int(repeat[1]) if repeat else [token]
                items += ["*"] * (14 - len(items))
                offset = 1 if keyword == "COMPDATL" else 0
                if items[7 + offset] == "*" and items[9 + offset] == "*":
                    wells.add(items[0].strip("'"))

    scan(Path(deck).resolve())
    return sorted(wells)
