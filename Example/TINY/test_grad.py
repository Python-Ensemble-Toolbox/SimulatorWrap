import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patheffects as patheffects
from matplotlib.ticker import FuncFormatter

from datetime import datetime
from subsurface.multphaseflow.jutul_darcy import JutulDarcy
from misc.structures import PETDataFrame, PETStateArray


# --------------------------------------------------------------------------- #
# Shared plotting setup
# --------------------------------------------------------------------------- #
NX, NY, NZ = 10, 10, 2

#: Well heads as 0-based (i, j) cell indices, from `include/Schdl.sch`.
#: Every well perforates both layers.
WELLS = {
    'INJ1': (0, 0), 'INJ2': (4, 0), 'INJ3': (9, 0),
    'PRO1': (0, 9), 'PRO2': (4, 9), 'PRO3': (9, 9),
}

INJECTOR_COLOUR = '#2f8fd4'
PRODUCER_COLOUR = '#d1495b'
DIVERGING_CMAP = 'coolwarm'


def _perforated_cells():
    """Flat (Fortran-order) indices of every perforated cell, both layers."""
    return np.array(sorted(
        i + j * NX + k * NX * NY
        for i, j in WELLS.values()
        for k in range(NZ)
    ))


def _well_colour(name):
    return PRODUCER_COLOUR if name.startswith('PRO') else INJECTOR_COLOUR


def _draw_wells(ax, highlight=None):
    """Overlay well heads; `highlight` is drawn filled to mark the objective well."""
    for name, (i, j) in WELLS.items():
        focus = name == highlight
        ax.plot(
            i, j, marker='o', linestyle='none', markersize=9 if focus else 6.5,
            markerfacecolor=_well_colour(name) if focus else 'none',
            markeredgecolor=_well_colour(name),
            markeredgewidth=2.0 if focus else 1.5, zorder=4,
        )
        if focus:
            # Label below the marker for wells in the upper half, so it never
            # collides with the panel title.
            above = j < NY / 2
            ax.annotate(
                name, (i, j), textcoords='offset points',
                xytext=(0, 11 if above else -19),
                ha='center', fontsize=8, fontweight='bold',
                color=_well_colour(name), zorder=5,
                # Halo keeps the label legible over saturated cells.
                path_effects=[patheffects.withStroke(linewidth=2.4,
                                                     foreground='white')],
            )


def _style_map_axes(ax):
    """Cell-index ticks and faint cell boundaries for a map panel."""
    ax.set_xticks(np.arange(0, NX, 2))
    ax.set_xticklabels(np.arange(1, NX + 1, 2), fontsize=8)
    ax.set_yticks(np.arange(0, NY, 2))
    ax.set_yticklabels(np.arange(1, NY + 1, 2), fontsize=8)
    ax.set_xticks(np.arange(-0.5, NX, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NY, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.4, alpha=0.55)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b0b0b0')
        spine.set_linewidth(0.8)


def _draw_map(ax, field, vmin, vmax, title):
    """
    Draw one 2-D layer with (i, j) running right and up.

    The colour range spans the actual data, `[vmin, vmax]`, so the norm is
    linear and the colorbar equispaced.
    """
    if vmin == vmax:                      # flat field: avoid a solid panel
        vmin, vmax = vmin - 0.5, vmax + 0.5
    im = ax.imshow(
        field.T, cmap=DIVERGING_CMAP, vmin=vmin, vmax=vmax,
        origin='lower', extent=(-0.5, NX - 0.5, -0.5, NY - 0.5),
        interpolation='nearest',
    )
    ax.set_title(title, fontsize=9.5, pad=6)
    _style_map_axes(ax)
    return im


def test_finite_difference_gradient(run_fda=True, run_adjoint=True, folder='TEST/FINITE_DIFF'):
    
    datapoint = 'WOPR:PRO2'
    date = datetime(2032, 12, 14)
    options = {
        'parallel': 5,
        'datatype': [datapoint],
        'reporttype': 'dates',
        'reportpoint': [date],
        'runfile': 'RUNFILE.mako',
        'startdate': datetime(2022, 1, 1),
        'adjoint_pbar': False,
        'adjoints': {'WOPR': 
            {'steps': [date], 'wellID': 'PRO2', 'parameters': 'permx'}
        },
        'perm_copied': True, 
    }

    os.makedirs(folder, exist_ok=True)

    # Load PERMX 
    permx = np.load('PERMX.npy')

    reps = 0.01 # Relative perturbation size for finite difference approximation (0.5%)
    if run_fda: 
        # Calculate finite difference gradient
        kw = dict(options)
        kw.pop('adjoints')

        # Pertubed input
        permx_p = np.tile(permx[:, None], (1, permx.size))
        permx_m = np.tile(permx[:, None], (1, permx.size))
        for i in range(permx.size):
            permx_p[i, i] += reps*permx[i]
            permx_m[i, i] -= reps*permx[i]

        # Run simulator on perturbed inputs
        simulator = JutulDarcy(kw)
        inputs_p  = [{'log_permx': np.log(permx_p[:, i])} for i in range(permx.size)]
        inputs_m  = [{'log_permx': np.log(permx_m[:, i])} for i in range(permx.size)]
        results_p = simulator(inputs_p)
        results_m = simulator(inputs_m)

        # Convert results to PETDataFrame and save
        results_p = PETDataFrame.merge_dataframes(results_p)
        results_m = PETDataFrame.merge_dataframes(results_m)
        results_p.to_pickle(os.path.join(folder, 'results_p.pkl'))
        results_m.to_pickle(os.path.join(folder, 'results_m.pkl'))
    
    if run_adjoint:
        # Run simulator with adjoint to get gradient
        simulator = JutulDarcy(options)
        inputs = [{'log_permx': np.log(permx)}]
        results, adjoint = simulator(inputs)
        adjoint.to_pickle(os.path.join(folder, 'adjoint.pkl'))
    
    # Analyze results
    results_p = PETDataFrame.from_pickle(os.path.join(folder, 'results_p.pkl')).loc[date, datapoint]
    results_m = PETDataFrame.from_pickle(os.path.join(folder, 'results_m.pkl')).loc[date, datapoint]
    grad_fda = (results_p - results_m)/(2*reps*permx)  # Finite difference approximation of gradient
    grad_adj = PETDataFrame.from_pickle(os.path.join(folder, 'adjoint.pkl')).loc[date, (datapoint, 'permx')]

    grad_fda = np.asarray(grad_fda, dtype=float)
    grad_adj = np.asarray(grad_adj, dtype=float)

    grad_fda_grid = grad_fda.reshape((NX, NY, NZ), order='F')
    grad_adj_grid = grad_adj.reshape((NX, NY, NZ), order='F')
    grad_fda_vec = grad_fda_grid.flatten(order='F')
    grad_adj_vec = grad_adj_grid.flatten(order='F')

    well = datapoint.split(':')[-1]
    perforated = _perforated_cells()
    unperforated = np.setdiff1d(np.arange(grad_fda_vec.size), perforated)

    # ------------------------------------------------------------------ #
    # Layer maps: finite difference | adjoint | difference, one row each,
    # with a parity plot and a summary panel on the right.
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(14.5, 7.4), dpi=140, constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.25])

    for layer in range(NZ):
        fda_layer = grad_fda_grid[:, :, layer]
        adj_layer = grad_adj_grid[:, :, layer]
        diff_layer = adj_layer - fda_layer

        # Finite difference and adjoint share a scale so they are comparable
        # by eye; the difference gets its own, or it would look uniformly blank.
        vmin = min(fda_layer.min(), adj_layer.min())
        vmax = max(fda_layer.max(), adj_layer.max())
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        peak = max(abs(vmin), abs(vmax))
        dmax = np.abs(diff_layer).max()

        ax_fda = fig.add_subplot(grid[layer, 0])
        ax_adj = fig.add_subplot(grid[layer, 1])
        ax_dif = fig.add_subplot(grid[layer, 2])

        _draw_map(ax_fda, fda_layer, vmin, vmax, 'Finite difference')
        im_shared = _draw_map(ax_adj, adj_layer, vmin, vmax, 'Adjoint')
        share = 100.0 * dmax / peak if peak else 0.0
        im_diff = _draw_map(
            ax_dif, diff_layer, -dmax, dmax,
            f'Difference  (peak {share:.1f}% of signal)',
        )

        for ax in (ax_fda, ax_adj, ax_dif):
            _draw_wells(ax, highlight=well)
        ax_fda.set_ylabel(f'Layer {layer + 1}\n$j$', fontsize=9)
        if layer == NZ - 1:
            for ax in (ax_fda, ax_adj, ax_dif):
                ax.set_xlabel('$i$', fontsize=9)

        cb = fig.colorbar(im_shared, ax=[ax_fda, ax_adj], fraction=0.046,
                          pad=0.02, shrink=0.92)
        cb.ax.tick_params(labelsize=7.5)
        cb_d = fig.colorbar(im_diff, ax=ax_dif, fraction=0.046, pad=0.04,
                            shrink=0.92)
        cb_d.ax.tick_params(labelsize=7.5)

    # ------------------------------------------------------------------ #
    # Parity plot: every cell, adjoint against finite difference.
    # ------------------------------------------------------------------ #
    ax_parity = fig.add_subplot(grid[0, 3])
    ax_parity.scatter(
        grad_fda_vec[unperforated], grad_adj_vec[unperforated], s=26,
        facecolor='#8fb8de', edgecolor='#33628f', linewidth=0.5,
        label=f'other cells ({unperforated.size})', zorder=2,
    )
    ax_parity.scatter(
        grad_fda_vec[perforated], grad_adj_vec[perforated], s=52, marker='D',
        facecolor=PRODUCER_COLOUR, edgecolor='white', linewidth=0.8,
        label=f'Well cells ({perforated.size})', zorder=3,
    )
    span = max(np.abs(grad_fda_vec).max(), np.abs(grad_adj_vec).max()) * 1.08
    ax_parity.plot([-span, span], [-span, span], color='#555555',
                   linewidth=1.0, linestyle='--', zorder=1, label='1:1')
    ax_parity.set_xlim(-span, span)
    ax_parity.set_ylim(-span, span)
    ax_parity.set_aspect('equal', adjustable='box')
    ax_parity.set_xlabel('Finite difference', fontsize=9)
    ax_parity.set_ylabel('Adjoint', fontsize=9)
    ax_parity.set_title(f'Agreement per cell (all {grad_fda_vec.size})',
                        fontsize=9.5, pad=6)
    ax_parity.tick_params(labelsize=8)
    ax_parity.grid(True, linewidth=0.5, alpha=0.35)
    ax_parity.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    for spine in ax_parity.spines.values():
        spine.set_edgecolor('#b0b0b0')

    # ------------------------------------------------------------------ #
    # Summary panel.
    # ------------------------------------------------------------------ #
    denom = np.linalg.norm(grad_adj_vec)
    rel_all = np.linalg.norm(grad_fda_vec - grad_adj_vec) / denom
    rel_perf = (np.linalg.norm(grad_fda_vec[perforated] - grad_adj_vec[perforated])
                / np.linalg.norm(grad_adj_vec[perforated]))
    slope = float(np.dot(grad_fda_vec, grad_adj_vec) / np.dot(grad_fda_vec, grad_fda_vec))

    ax_text = fig.add_subplot(grid[1, 3])
    ax_text.axis('off')
    ax_text.text(
        0.0, 0.97,
        '\n'.join([
            f'objective      {datapoint}',
            f'date           {date:%d %b %Y}',
            f'FD step        {100 * reps:.1f}% of PERMX',
            '',
            f'relative error       {rel_all:7.2%}',
            f'  at perforations    {rel_perf:7.2%}',
            f'best-fit slope       {slope:7.4f}',
            '',
            f'||adjoint||          {denom:9.3f}',
            f'||finite diff||      {np.linalg.norm(grad_fda_vec):9.3f}',
        ]),
        transform=ax_text.transAxes, va='top', ha='left',
        family='monospace', fontsize=8.5, linespacing=1.6,
    )

    fig.suptitle(
        f'Adjoint gradient $\\partial$({datapoint})$/\\partial$PERMX '
        'vs. central finite differences',
        fontsize=12.5, fontweight='bold',
    )
    fig.savefig(os.path.join(folder, 'fda_vs_adjoint_gradient.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    relative_error = np.linalg.norm(grad_fda_vec - grad_adj_vec) / np.linalg.norm(grad_adj_vec)
    print(f"Relative error between FDA and adjoint gradients: {relative_error:.4e}")
    print(f"Norm of FDA gradient: {np.linalg.norm(grad_fda_vec):.4e}")
    print(f"Norm of adjoint gradient: {np.linalg.norm(grad_adj_vec):.4e}")


    rtol = grad_adj_vec.size * np.finfo(float).eps * 100  # Relative tolerance scaled by size of gradient vector
    atol = 1e-2 * np.linalg.norm(grad_adj_vec)  #
    print(f"Using rtol={rtol:.4e} and atol={atol:.4e} for np.testing.assert_allclose")
    np.testing.assert_allclose(grad_fda_vec, grad_adj_vec, rtol=rtol, atol=atol)
    


def test_sens_matrix_of_log_permx(run=True, folder='TEST'):

    datapoint = 'WOPR:PRO2'
    date = datetime(2032, 12, 14)
    options = {
        'parallel': 5,
        'datatype': [datapoint],
        'reporttype': 'dates',
        'reportpoint': [date],
        'runfile': 'RUNFILE.mako',
        'startdate': datetime(2022, 1, 1),
        'adjoint_pbar': False,
        'adjoints': {'WOPR': 
            {'steps': [date], 'wellID': 'PRO2', 'parameters': 'log_permx'}
        },
    }

    os.makedirs(folder, exist_ok=True)

    ne = 10_000
    if run: 
        np.random.seed(29_01_1983)

        pinfo = {
            'nx': 10,
            'ny': 10,
            'nz': 2,
            'vario': ['sph', 'sph'],
            'mean': 200*[4.0],
            'variance': [1.0, 1.0],
            'corr_length': [10.0, 10.0],
            'aniso': [1.0, 1.0],
            'angle': [0.0, 0.0],
        }
        prior_log_permx_ensemble = PETStateArray.generate_from_prior_info(
            prior_info = {'log_permx': pinfo},
            ne=ne,
            save=False
        )
        np.save(os.path.join(folder, 'prior_log_permx_ensemble.npy'), prior_log_permx_ensemble)

        # Run simulator on ensemble
        simulator = JutulDarcy(options)
        inputs = [{'log_permx': prior_log_permx_ensemble[:, i]} for i in range(ne)]
        results, adjoint = simulator(inputs)
        results = PETDataFrame.merge_dataframes(results)
        adjoint = PETDataFrame.merge_dataframes(adjoint)
        results.to_pickle(os.path.join(folder, 'results.pkl'))
        adjoint.to_pickle(os.path.join(folder, 'adjoint.pkl'))

    
    # Analyze results
    col = datapoint
    idx = date
    results = PETDataFrame.from_pickle(os.path.join(folder, 'results.pkl')).loc[idx, col]
    adjoint = PETDataFrame.from_pickle(os.path.join(folder, 'adjoint.pkl')).loc[idx, (col, 'log_permx')]

    nx = 200
    ny = 1
    enX = np.load(os.path.join(folder, 'prior_log_permx_ensemble.npy'))
    enY = results[np.newaxis, :]
    enG = adjoint[np.newaxis, :, :]

    assert enX.shape == (nx, ne)
    assert enY.shape == (ny, ne)
    assert enG.shape == (ny, nx, ne)

    # Compute sensitivity matrix using ensemble gradients
    P = (np.eye(ne) - np.ones((ne,ne))/ne)/np.sqrt(ne-1)
    A = enX @ P
    Y = enY @ P
    Gbar = np.mean(enG, axis=-1)

    Cyx = Y @ A.T
    GbarCxx = Gbar @ A @ A.T

    # -------------------------------------------------------------------
    # Stein's lemma tells us that:
    # E[G]Cxx = Cyx   --->   Gbar @ A @ A.T ≈ Y @ A.T (for large ne)
    # -------------------------------------------------------------------

    # Loop over different ensemble sizes and compute norms of Cyx, GbarCxx, and their difference
    Cxy_norm = []
    GbarCxx_norm = []
    err = []
    ens = np.logspace(1, np.log10(ne), num=10, dtype=int)
    for n in ens:
        P_n = (np.eye(n) - np.ones((n,n))/n)/np.sqrt(n-1)
        A_n = enX[:, :n] @ P_n
        Y_n = enY[:, :n] @ P_n
        Gbar_n = np.mean(enG[:, :, :n], axis=-1)
        
        GbarCxx_n = Gbar_n @ A_n @ A_n.T
        Cyx_n = Y_n @ A_n.T

        err.append(np.linalg.norm(GbarCxx_n-Cyx_n))
        Cxy_norm.append(np.linalg.norm(Cyx_n))
        GbarCxx_norm.append(np.linalg.norm(GbarCxx_n))

    ens = np.asarray(ens, dtype=float)
    Cxy_norm = np.asarray(Cxy_norm)
    GbarCxx_norm = np.asarray(GbarCxx_norm)
    err = np.asarray(err)
    rel_err = err / GbarCxx_norm[-1] # Relative error normalized by the last GbarCxx norm


    # ------------------------------------------------------------------ #
    # Stein's lemma predicts GbarCxx -> Cyx. The left panel shows both
    # quantities settling; the right panel shows what actually matters, the
    # relative gap between them, against the Monte-Carlo rate it should follow.
    # ------------------------------------------------------------------ #
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, (ax_norm, ax_err) = plt.subplots(
            1, 2, figsize=(12, 4.6), dpi=140, constrained_layout=True,
        )

        line_kw = dict(linewidth=2.2, markersize=6.5, markerfacecolor='white',
                       markeredgewidth=1.5)
        ax_norm.plot(ens, Cxy_norm, color='#1f77b4', marker='o',
                     label=r'$\|C_{yx}\|$  (from the ensemble)', **line_kw)
        ax_norm.plot(ens, GbarCxx_norm, color='#2ca02c', marker='s',
                     label=r'$\|\bar{G}C_{xx}\|$  (from the adjoints)', **line_kw)
        ax_norm.set_xscale('log')
        ax_norm.set_xlabel('Ensemble size  $N_e$', fontsize=10.5)
        ax_norm.set_ylabel(r'$L_2$-norm', fontsize=10.5)
        ax_norm.set_title('Both sides of Stein\'s identity', fontsize=11.5, pad=8)
        ax_norm.set_ylim(0, None)
        ax_norm.legend(fontsize=9.5, loc='best', framealpha=0.92)

        ax_err.plot(ens, rel_err, color='#d62728', marker='^',
                    label=r'$\|\bar{G}C_{xx} - C_{yx}\|\,/\,\|C_{yx}\|$',
                    **line_kw)
        # Monte-Carlo reference anchored on the first point.
        ax_err.plot(ens, rel_err[0]*np.sqrt(ens[0]/ens), color='#888888',
                    linewidth=1.4, linestyle='--',
                    label=r'$\propto 1/\sqrt{N_e}$')
        ax_err.set_xscale('log')
        ax_err.set_yscale('log')
        ax_err.set_xlabel('Ensemble size  $N_e$', fontsize=10.5)
        ax_err.set_ylabel('Relative error', fontsize=10.5)
        # Percentages read faster than 6x10^0 on a log axis; label the minor
        # ticks too, since the range can span well under a decade.
        pct = FuncFormatter(lambda v, _: f'{100 * v:g}%' if v > 0 else '')
        ax_err.yaxis.set_major_formatter(pct)
        ax_err.yaxis.set_minor_formatter(pct)
        ax_err.set_title('Gap between them', fontsize=11.5, pad=8)
        ax_err.legend(fontsize=9.5, loc='best', framealpha=0.92)
        ax_err.annotate(
            f'{rel_err[-1]:.1%} at $N_e={int(ens[-1])}$',
            xy=(ens[-1], rel_err[-1]), xytext=(-12, 16),
            textcoords='offset points', ha='right', fontsize=9,
            color='#d62728', fontweight='bold',
        )

        for ax in (ax_norm, ax_err):
            ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.45)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.25)
            ax.tick_params(labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle(
            f"Ensemble sensitivity of {datapoint} to log-PERMX\n"
            r"Stein: $\mathbb{E}[G]\,C_{xx} = C_{yx}$ for Gaussian inputs",
            fontsize=12.5, fontweight='bold',
        )
        fig.savefig(os.path.join(folder, 'sensitivity_matrix_convergence.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    


if __name__ == "__main__":
    test_finite_difference_gradient(run_fda=False, run_adjoint=False, folder='TEST/TEMP')
    #test_sens_matrix_of_log_permx(run=False, folder='TEST/SENS')