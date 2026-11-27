# Subsurface
Repository containing some standard simulation wrappers. For a detailed description of how to build the wrappers,
see the documentation for PET.

## Installation
Clone the repository and install it in editable mode by running
```bash
python3 -m pip install -e .
```

## JutulDarcy weighted adjoints for PET (LowRankGN and SVGD)

Pass the existing `subsurface.multphaseflow.jutul_darcy.JutulDarcy(options)`
to PET. Use a Mako `runfile`, `adjoint_mode="sensitivities"` (the default),
and omit `adjoints`. Ordinary forecasts then return just predictions. The
existing `adjoints` configuration remains available for standalone
per-observation gradients.

`run_vjp(state, weights)` accepts PET's report-by-datatype frame. An
observed cell contains the k coefficients of that datum, a missing cell
is `None`. It runs the centre once per call, shares its ministeps and
adjoint storage across the columns, and returns `{name: array(k, n)}` in
the coordinates of each state field. There is no cross-call cache. The
wrapper checks each weighted objective against the direct forward summary.

For SVGD, set `weighted_adjoint={"parameters": ["log_permx"]}` instead.
PET sets `sim.adjoint_weights` to a callable `(forward_frame, member_index)`
returning one scalar weight per observed cell (missing weights are zero).
`run_fwd_sim` returns `(output, gradient_series)` from that member's live
trajectory, with no second forward solve. Match the gradient coordinate
to the state: `permx` differentiates mD, `log_permx` its natural logarithm.

Both entry points support METRIC well observations on report steps: `WOPR`,
`WWPR`, `WGPR`, `WLPR`, `WWIR` (Sm³/day) and `WOPT`, `WWPT`, `WGPT`,
`WLPT` (Sm³). The batched VJP's supported controls are
direct `poro` (fraction), `permx`, `permy`, `permz` (mD) and `log_permx`
etc. (natural log of mD). Use Cartesian deck order, full grid or processed
active cells. The permeability gradient includes the wrapper's existing
well-index and copied-axis chain rules. Specify `perm_copied=False` for
independent axes or `True` for three identical copies when the deck fields
alone are ambiguous. Inputs must render directly: clipping outside bounds,
fault factors, and other template dependencies need their own chain rule.

`check_chain_rule(state, weights=None)` tells whether the derivative resolves
every term of the chain rule from the state to the weighted sums. It perturbs
each variable, renders the deck again and lets JutulDarcy set the case up
without simulating, then compares the rendered text and the arrays of the two
cases: data domains, parameters and initial state of every model. Only the
variable's own field may move, the reservoir parameters JutulDarcy computes
from it, the axes `perm_copied` adds and the well indices
`well_index_from_perm` differentiates. `run_vjp` runs this part itself before
its first batch. It raises on a term it would miss, and on a term the options
add for an axis or a well index that does not move; this settles the cases the
detection from the deck fields cannot, such as a uniform `PERMX`. The returned
report lists the tied axes (`ties`) and the derived parameters (`derived`).

With `weights`, the gradients are also compared with central differences of
the weighted sums along an alternating and a uniform direction, at two step
sizes: four simulations per variable and direction. A mismatch raises. A row
with `ok = None` could not be judged because the finite differences change
too much with the step.

**Forward summary unit change:** field and well surface totals (including
`WWIT` and `FLPT`) are now converted from SI volumes. Previously JutulDarcy
0.3.11's METRIC/FIELD cumulative outputs were 86,400 times too large. Rates
are reported per day for METRIC/FIELD, per hour for LAB, and per second for
SI. Do not rescale PET observations or gradients to compensate for the old
output. Recheck any prior workaround before comparing stored forecasts.

With the wrapper's Julia environment active, run `python -m pytest -q
tests/test_jutul_darcy.py tests/test_jutul_vjp.py` for both derivative
paths, the directional finite difference and the SI-volume regression.

## Examples
### PET LowRankGN: weighted Flow adjoints

Use `subsurface.multphaseflow.opm_adjoint.FlowAdjoint(options)` in place of
`flow(options)`. Give it the usual Flow options plus `adjoint_executable` (an
absolute executable path or command on PATH; default `flow_adjoint`), the
executable of [OPM/opm-adjoint](https://github.com/OPM/opm-adjoint) built
against the official OPM modules. Stock `flow` has no adjoint. No Python
dependency is added.

After `setup_fwd_run()`, `run_vjp(state, weights)` renders the centre once,
records a trajectory, and replays it serially for each nonzero seed column.
`weights` is PET's report-by-datatype frame: each observed cell holds a
length-k vector; `None` cells are skipped. Each returned `grad[name][j]` is
the derivative of the j-th weighted summary with respect to `state[name]`.
Forward members never record an archive. The temporary archive needs enough
disk for all substeps and is removed after the batch.

The weighted sums are handed to `flow_adjoint` as a JSON objective
(`--adjoint-config`): one `rate` term per observed value, active on the
substep that ends on its report. Gradients are read from the
`ADJOINT_GRADIENTS_{PORO,PERM}.csv` files and their JSON sidecar. Each batch
adds one objective-only replay, which fixes the substep lengths the terms
need.

- Use scalar `WOPR`, `WWPR` or `WGPR` for **producing wells** in Sm³/day. A
  report may be any time a simulator step ends on; the wrapper checks the
  objective against the recorded summary, which must hold every simulator
  step: `RPTONLY` in the deck is switched off for the recorded run.
- Use serial METRIC decks writing `INIT`, with independent `poro` (fraction),
  `permx`, `permy`, `permz` (mD), or `log_permx` etc. (natural log of mD), one
  value per active cell or per grid cell in natural order. The gradient has
  the shape of the state; inactive cells get zero. A permeability axis the
  deck ties to the variable's field by a constant factor (`COPY`, `MULTIPLY`)
  is found and added to its gradient. Anything else that moves with the
  state is refused: another keyword the template writes from it, a clipped
  or otherwise transformed field, an array Flow derives from it that
  opm-adjoint does not differentiate.
- Connection factors computed from the permeability and multisegment wells
  are differentiated by opm-adjoint. Two cases need changes that are not in
  its master yet:
  [OPM/opm-adjoint#2](https://github.com/OPM/opm-adjoint/pull/2) for exact
  gradients with producers on RESV control, and
  [OPM/opm-adjoint#3](https://github.com/OPM/opm-adjoint/pull/3) for wells
  that the schedule defines after the first report step.
- The same executable and solver flags drive both predictions and adjoints.
  opm-adjoint's own solver flags, such as `--adjoint-linear-solver=umfpack`,
  go in `sim_flag` with the others; the wrapper sets the archive, mode,
  objective and output flags itself and refuses them there.
  Use converged forward tolerances and check a directional derivative when
  moving to a new deck (below); matching objective values checks units and
  timing, not the accuracy of the backend derivative.

`check_chain_rule(state, weights=None)` tells whether the derivative resolves
every term of the chain rule from the state to the weighted sums. It perturbs
each variable, renders the deck again and lets Flow read it without
simulating, then compares the rendered text and Flow's `INIT` arrays with the
unperturbed ones. `run_vjp` runs this part itself before its first batch, and
raises on a term it would miss. The returned report lists the tied axes
(`ties`), the arrays Flow derives (`derived`) and what opm-adjoint
differentiates approximately (`approximate`: connection factors left for Flow
to compute from the permeability, of which it keeps the Kh term).

With `weights`, the gradients are also compared with central differences of
the weighted sums along an alternating and a uniform direction, at two step
sizes: four simulations per variable and direction. This is the check that
reaches dependencies inside Flow which its `INIT` arrays do not show. A
mismatch raises. A row with `ok = None` could not be judged because the
finite differences change too much with the step: the simulated rates are
not smooth in the state, which adaptive time stepping and loose solver
tolerances cause; fixed steps and tight tolerances make it conclusive.

Run the directional derivative tests with `FLOW_ADJOINT` pointing to the
executable and `OPM_ADJOINT_CASE` to `doc/tutorials/MODEL_1D_DEBUG` of
opm-adjoint: `python -m pytest -q tests/test_flow_vjp.py` (with PET on the
Python path).

### Forward Flow example
The example demonstrates usage of [OPM-flow](https://opm-project.org/). To reproduce, ensure that OPM-flow is installed and in the path. The full noteboox is [here](https://github.com/Python-Ensemble-Toolbox/SimulatorWrap/tree/main/Example), we employ the [3Spot example](https://github.com/Python-Ensemble-Toolbox/Examples/tree/main/3Spot) as a template for this example. All files are Example folder.


```python
# import the flow class
from subsurface.multphaseflow.opm import flow
# import datatime
import datetime as dt
# import numpy
import numpy as np
```

To initialllize the wrapper such that we get the needed outputs we must specify multiple inputs. This will typically be extracted from a config .toml file.


```python
# a dictionary containing relevant information
input_dict = {'parallel':2,
             'simoptions': [['sim_flag', '--tolerance-mb=1e-5 --parsing-strictness=low']],
             'sim_limit': 4,
             'reportpoint': [dt.datetime(1994,2,9,00,00,00),
                            dt.datetime(1995,1,1,00,00,00),
                            dt.datetime(1996,1,1,00,00,00),
                            dt.datetime(1997,1,1,00,00,00),
                            dt.datetime(1998,1,1,00,00,00),
                            dt.datetime(1999,1,1,00,00,00)],
            'reporttype': "dates",
            'datatype': [
                "fopt",
                "fgpt",
                "fwpt",
                "fwit",
                "rft:pro-1"],
             'runfile':'3well'}

# name of the runfile
filename = '3WELL'
```


```python
# Generate an instance of the simulator class
sim = flow(input_dict=input_dict,filename=filename)
```


```python
# Setup simulator
sim.setup_fwd_run(redund_sim=None)
```

The point of the simulator wrapper in PET is to generate the simulation response for an ensemble of parameters. The mako template file needs to render a DATA file for each uncertain parameter. Hence, the syntax of the mako file need to match the test one wants to run. It is up to the user to specify this file. In the 3Spot case, the uncertain parameter consists of the bhp controll for the wells. In the wrapper this is a dictionary with keys matching the variablename in the mako file.


```python
state = {'injbhp':np.array([280,245]),
         'prodbhp':np.array([110])}
```


```python
# we can now run the flow simulator
pred = sim.run_fwd_sim(state,member_i=0,del_folder=True)
```


```python
pred
```


    [{
        'fopt': array(1.34704602),
        'fgpt': array(0.),
        'fwpt': array(0.00086854),
        'fwit': array(2.7143116),
        'rft:pro-1': array(nan)},
        {'fopt': array(277.52658081),
        'fgpt': array(0.),
        'fwpt': array(0.77613616),
        'fwit': array(480.96279907),
        'rft:pro-1': array([161.88829, 162.72838], dtype='>f4')},
        {'fopt': array(573.01849365),
        'fgpt': array(0.),
        'fwpt': array(1.46081805),
        'fwit': array(923.75933838),
        'rft:pro-1': array(nan)},
        {'fopt': array(866.71929932),
        'fgpt': array(0.),
        'fwpt': array(2.07612634),
        'fwit': array(1355.08740234),
        'rft:pro-1': array([160.21707, 161.05704], dtype='>f4')},
        {'fopt': array(1156.9161377),
        'fgpt': array(0.),
        'fwpt': array(2.6172893),
        'fwit': array(1787.93688965),
        'rft:pro-1': array(nan)},
        {'fopt': array(1445.58898926),
        'fgpt': array(0.),
        'fwpt': array(3.12104678),
        'fwit': array(2229.38110352),
        'rft:pro-1': array([159.58162, 160.4215 ], dtype='>f4')
        }]


```python
# we can make simple plot using matplotlib
import matplotlib.pyplot as plt
# Display plots inline in Jupyter
%matplotlib inline
```


```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fopt'].flatten() for el in pred])));
plt.title('Field Oil Production Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```

<h1 align="center">
<img src="./Example/README_11_0.png">
</h1><br>


```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fgpt'].flatten() for el in pred])));
plt.title('Field Gas Production Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```


<h1 align="center">
<img src="./Example/README_12_0.png">
</h1><br>



```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fwpt'].flatten() for el in pred])));
plt.title('Field Water Production Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```


<h1 align="center">
<img src="./Example/README_13_0.png">
</h1><br>



```python
plt.figure(figsize=(10, 6));
plt.plot(input_dict['reportpoint'],np.concatenate(np.array([el['fwit'].flatten() for el in pred])));
plt.title('Field Water Injection Total');
plt.xlabel('Report dates');
plt.ylabel('STB');
```


<h1 align="center">
<img src="./Example/README_14_0.png">
</h1><br>


```python
import matplotlib.dates as mdates

plt.figure(figsize=(10, 6))
ax = plt.gca()
for reportpoint, rft_values in zip(input_dict['reportpoint'], [el['rft:pro-1'] for el in pred]):
    rft_values = np.asarray(rft_values).flatten()
    if rft_values.size == 1 and np.isnan(rft_values[0]):
        continue
    rft_values = rft_values[~np.isnan(rft_values)]
    if rft_values.size == 0:
        continue
    ax.plot([reportpoint] * rft_values.size, rft_values, color='C0')
ax.set_xlim(dt.datetime(input_dict['reportpoint'][0].year, 1, 1), max(input_dict['reportpoint']) + dt.timedelta(days=30))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title('RFT PRO-1');ax.set_xlabel('Report dates');ax.set_ylabel('Bar');
```

<h1 align="center">
<img src="./Example/README_15_0.png">
</h1><br>
