"""Descriptive description."""
from selectors import SelectSelector

from subsurface.multphaseflow.opm import flow
from importlib import import_module
import datetime as dt
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
from misc import ecl, grdecl
import shutil
import glob
from subprocess import Popen, PIPE
import mat73
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
from scipy.special import jv  # Bessel function of the first kind
from scipy.integrate import quad
from scipy.special import j0
from mako.lookup import TemplateLookup
from mako.runtime import Context
#import cProfile
#import pstats

# from pylops import avo
from pylops.utils.wavelets import ricker
from pylops.signalprocessing import Convolve1D
import sys
#from PyGRDECL.GRDECL_Parser import GRDECL_Parser  # https://github.com/BinWang0213/PyGRDECL/tree/master
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from geostat.decomp import Cholesky
#from simulator.eclipse import ecl_100
from CoolProp.CoolProp import PropsSI  # http://coolprop.org/#high-level-interface-example

class mixIn_multi_data():

    def find_cell_centre(self, grid):
        # Find indices where the boolean array is True
        indices = np.where(grid['ACTNUM'])

        coord = grid['COORD']
        zcorn = grid['ZCORN']

        c, a, b = indices
        # Calculate xt, yt, zt
        xb = 0.25 * (coord[a, b, 0, 0] + coord[a, b + 1, 0, 0] + coord[a + 1, b, 0, 0] + coord[a + 1, b + 1, 0, 0])
        yb = 0.25 * (coord[a, b, 0, 1] + coord[a, b + 1, 0, 1] + coord[a + 1, b, 0, 1] + coord[a + 1, b + 1, 0, 1])
        zb = 0.25 * (coord[a, b, 0, 2] + coord[a, b + 1, 0, 2] + coord[a + 1, b, 0, 2] + coord[a + 1, b + 1, 0, 2])

        xt = 0.25 * (coord[a, b, 1, 0] + coord[a, b + 1, 1, 0] + coord[a + 1, b, 1, 0] + coord[a + 1, b + 1, 1, 0])
        yt = 0.25 * (coord[a, b, 1, 1] + coord[a, b + 1, 1, 1] + coord[a + 1, b, 1, 1] + coord[a + 1, b + 1, 1, 1])
        zt = 0.25 * (coord[a, b, 1, 2] + coord[a, b + 1, 1, 2] + coord[a + 1, b, 1, 2] + coord[a + 1, b + 1, 1, 2])

        # Calculate z, x, and y positions
        z = (zcorn[c, 0, a, 0, b, 0] + zcorn[c, 0, a, 1, b, 0] + zcorn[c, 0, a, 0, b, 1] + zcorn[c, 0, a, 1, b, 1] +
             zcorn[c, 1, a, 0, b, 0] + zcorn[c, 1, a, 1, b, 0] + zcorn[c, 1, a, 0, b, 1] + zcorn[c, 1, a, 1, b, 1]) / 8
        denom = zt - zb
        valid = denom > 0

        x = np.copy(xb).astype(float)
        y = np.copy(yb).astype(float)

        if np.any(valid):
            frac = np.empty_like(z, dtype=float)
            frac[valid] = (z[valid] - zb[valid]) / denom[valid]
            x[valid] = xb[valid] + (xt[valid] - xb[valid]) * frac[valid]
            y[valid] = yb[valid] + (yt[valid] - yb[valid]) * frac[valid]

        #x = xb + (xt - xb) * (z - zb) / (zt - zb)
        #y = yb + (yt - yb) * (z - zb) / (zt - zb)

        cell_centre = [x, y, z]
        return cell_centre
        

    def get_seabed_depths(self, file_path):
        # Read the data while skipping the header comments
        # We'll assume the header data ends before the numerical data
        # The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
        water_depths = pd.read_csv(file_path, comment='#', sep=r'\s+',
                                   header=None)  # delim_whitespace=True, header=None)

        # Give meaningful column names:
        water_depths.columns = ['x', 'y', 'z', 'column', 'row']

        return water_depths

    def measurement_locations(self, grid, water_depth, pad=1500, dxy=3000, well_coord = None, r0 = 5000):

        # Determine the size of the measurement area as defined by the field extent
        cell_centre = self.find_cell_centre(grid)
        x_min = np.min(cell_centre[0])
        x_max = np.max(cell_centre[0])
        y_min = np.min(cell_centre[1])
        y_max = np.max(cell_centre[1])

        x_min -= pad
        x_max += pad
        y_min -= pad
        y_max += pad

        x_span = x_max - x_min
        y_span = y_max - y_min

        nx = int(np.ceil(x_span / dxy))
        ny = int(np.ceil(y_span / dxy))

        x_vec = np.linspace(x_min, x_max, nx)
        y_vec = np.linspace(y_min, y_max, ny)
        x, y = np.meshgrid(x_vec, y_vec)

        # allow for finer measurement grid around injection well
        if well_coord is not None:
            # choose center point and radius for area with finer measurement grid
            # well_coord should be [x_coordinate, y_coordinate] instead of grid indices
            # Example: [2700.0, 1000.0] for Smeaheia instead of [34, 64]
            xc = well_coord[0]
            yc = well_coord[1]

            dxy_fine = round(dxy/2)

            # Fine grid covering bounding box of the circle (clamped to domain)
            fx_min = max(x_min, xc - r0)
            fx_max = min(x_max, xc + r0)
            fy_min = max(y_min, yc - r0)
            fy_max = min(y_max, yc + r0)

            pts_coarse = np.column_stack((x.ravel(), y.ravel()))

            if fx_max > fx_min and fy_max > fy_min:
                nfx = int(np.ceil((fx_max - fx_min) / dxy_fine)) + 1
                nfy = int(np.ceil((fy_max - fy_min) / dxy_fine)) + 1
                x_fine = np.linspace(fx_min, fx_max, nfx)
                y_fine = np.linspace(fy_min, fy_max, nfy)
                xf, yf = np.meshgrid(x_fine, y_fine)
                pts_fine = np.column_stack((xf.ravel(), yf.ravel()))

                # Keep only fine points inside the circle
                d2 = (pts_fine[:, 0] - xc) ** 2 + (pts_fine[:, 1] - yc) ** 2
                mask_inside = d2 <= r0 ** 2
                pts_fine_inside = pts_fine[mask_inside]

                # remove the coarse points inside the circle
                d2 = (pts_coarse[:, 0] - xc) ** 2 + (pts_coarse[:, 1] - yc) ** 2
                mask_inside = d2 <= r0 ** 2
                pts_coarse = pts_coarse[~mask_inside]

                # Combine and remove duplicates by rounding to a tolerance or using a structured array
                # Use tolerance based on the smaller spacing
                tol = min(dxy, dxy_fine) * 1e-3
                all_pts = np.vstack((pts_coarse, pts_fine_inside))

                # Round coordinates to avoid floating point duplicates then use np.unique
                # Determine digits to round so differences smaller than tol collapse
                digits = max(0, int(-np.floor(np.log10(tol))))
                all_pts_rounded = np.round(all_pts, digits)
                uniq_pts = np.unique(all_pts_rounded, axis=0)
                x = uniq_pts[:, 0]
                y = uniq_pts[:, 1]


        pos = {'x': x.flatten(), 'y': y.flatten()}

        # Seabed map or water depth scalar depending on input
        if isinstance(water_depth, float) or isinstance(water_depth, int):
            pos['z'] = np.ones_like(pos['x']) * water_depth
        else:
            pos['z'] = griddata((water_depth['x'], water_depth['y']),
                                np.abs(water_depth['z']), (pos['x'], pos['y']),
                                method='nearest')  # z is positive downwards
        return pos

    def filter_rporv(self, arr, cutoff = None, method='fixed', threshold=None, percentile=95.5, k=3.0,top_frac=0.01, ratio_thresh=10.0):

        m = ma.array(arr, copy=True)

        # convert object elements to floats if necessary
        if m.dtype == object:
            flat = np.array([getattr(x, 'value', x) for x in m.ravel()], dtype=float)
            m = ma.array(flat.reshape(m.shape), mask=ma.getmaskarray(m))

        data = m.compressed().astype(float)
        if data.size == 0:
            return m


        if cutoff is None:
            if method == 'fixed':
                if threshold is None:
                    raise ValueError("threshold required for method='fixed'")
                cutoff = float(np.asarray(threshold).item())
            elif method == 'percentile':
                cutoff = float(np.nanpercentile(data, percentile))
            elif method == 'sigma':
                cutoff = float(np.nanmean(data) + k * np.nanstd(data))
            elif method == 'median_relative':
                # threshold here is interpreted as a multiplicative factor; if None, use median_factor
                if threshold is None:
                    raise ValueError("threshold (median factor) required for method='median_relative'")
                factor = float(np.asarray(threshold).item())
                med = float(np.nanmedian(data))
                cutoff = med * factor
            else:
                raise ValueError("unknown method")


        # preserve original mask, set values > cutoff to 0.0 in the underlying data
        orig_mask = ma.getmaskarray(m)
        filled = m.filled(np.nan).astype(float)

        # Decide whether aquifer-like outliers exist
        n_top = max(1, int(np.ceil(top_frac * data.size)))
        sorted_vals = np.sort(data)
        top_vals = sorted_vals[-n_top:]
        median_bulk = float(np.nanmedian(sorted_vals[:-n_top]) if data.size > n_top else np.nanmedian(sorted_vals))
        mean_top = float(np.nanmean(top_vals))
        aquifer_present = False
        if np.isfinite(median_bulk) and median_bulk > 0:
            if (mean_top / median_bulk) >= ratio_thresh:
                aquifer_present = True
        # only remove cells, if very big
        if aquifer_present:
            print(f'Remove aquifer cells')
            filled[filled > cutoff] = 0.0

        res = ma.array(filled, mask=orig_mask)
        return res, cutoff


class flow_rock(flow):
    """
    Couple the OPM-flow simulator with a rock-physics simulator such that both reservoir quantities and petro-elastic
    quantities can be calculated. Inherit the flow class, and use super to call similar functions.
    """

    def __init__(self, input_dict=None, filename=None, options=None):
        super().__init__(input_dict)
        self._getpeminfo(input_dict)

        self.date_slack = None
        if 'date_slack' in input_dict:
            self.date_slack = int(input_dict['date_slack'])

        # If we want to extract, or evaluate, something uniquely from the ensemble specific run we can
        # run a user defined code to do this.
        self.saveinfo = None
        if 'savesiminfo' in input_dict:
            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(input_dict['savesiminfo'], list):
                self.saveinfo = input_dict['savesiminfo']
            else:
                self.saveinfo = [input_dict['savesiminfo']]

        self.scale = []

        # Store dynamic variables in case they are provided in the state
        self.state = None
        self.no_flow = False

    def _getpeminfo(self, input_dict):
        """
        Get, and return, flow and PEM modules
        """
        if 'pem' in input_dict:
            self.pem_input = {}
            for elem in input_dict['pem']:
                if elem[0] == 'model':  # Set the petro-elastic model
                    self.pem_input['model'] = elem[1]
                if elem[0] == 'depth':  # provide the npz of depth values
                    self.pem_input['depth'] = elem[1]
                if elem[0] == 'actnum':  # the npz of actnum values
                    self.pem_input['actnum'] = elem[1]
                if elem[0] == 'baseline':  # the time for the baseline 4D measurement
                    self.pem_input['baseline'] = elem[1]
                if elem[0] == 'vintage':
                    self.pem_input['vintage'] = elem[1]
                    if not type(self.pem_input['vintage']) == list:
                        self.pem_input['vintage'] = [elem[1]]
                if elem[0] == 'ntg':
                    self.pem_input['ntg'] = elem[1]
                if elem[0] == 'press_conv':
                    self.pem_input['press_conv'] = elem[1]
                if elem[0] == 'compaction':
                    self.pem_input['compaction'] = True
                if elem[0] == 'overburden':  # the npz of overburden values
                    self.pem_input['overburden'] = elem[1]
                if elem[0] == 'percentile':  # use for scaling
                    self.pem_input['percentile'] = elem[1]
                if elem[0] == 'phases':  # get the fluid phases
                    if isinstance(elem[1], list):
                        self.pem_input['phases'] = [str.upper(item) for item in elem[1]]
                    else:
                        phases = str.upper(elem[1])
                        self.pem_input['phases'] = phases.split()
                if elem[0] == 'grid':  # get the model grid
                    self.pem_input['grid'] = elem[1]
                if elem[0] == 'param_file':  # get model parameters required for pem
                    self.pem_input['param_file'] = elem[1]


            pem = getattr(import_module('subsurface.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)
            if not hasattr(self.pem, 'input'):
                self.pem.input = self.pem_input

        else:
            self.pem = None

    def _get_pem_input(self, type, time=None):
        if self.no_flow:  # get variable from state
            if any(type.lower() in key for key in self.state.keys()) and time > 0:
                data = self.state[type.lower()+'_'+str(time)]
                mask = np.zeros(data.shape, dtype=bool)
                return np.ma.array(data=data, dtype=data.dtype,
                              mask=mask)
            else:  # read parameter from file
                param_file = self.pem_input['param_file']
                npzfile = np.load(param_file)
                parameter = npzfile[type]
                npzfile.close()
                data = parameter[:,self.ensemble_member]
                mask = np.zeros(data.shape, dtype=bool)
                return np.ma.array(data=data, dtype=data.dtype,
                                   mask=mask)
        else:  # get variable of parameter from flow simulation
            return self.ecl_case.cell_data(type,time)

    def _recover_missing_well_summary_data(self, member):
        """Backfill missing well-summary values when key separators differ (space vs colon)."""
        try:
            case = ecl.EclipseCase('En_' + str(member) + os.sep + self.file)
        except Exception:
            return

        for prim_ind in self.l_prim:
            if self.true_prim[0] == 'days':
                time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                    dt.timedelta(days=self.true_prim[1][prim_ind])
            else:
                time = self.true_prim[1][prim_ind]

            for key in self.all_data_types:
                if self.pred_data[prim_ind][key] is not None:
                    continue
                # Only attempt recovery for well-summary style keys (e.g. "WBHP INJ0").
                if len(key.split(' ')) != 2 or 'rft_' in key.lower():
                    continue

                for cand in (key, key.replace(' ', ':'), key.replace(':', ' ')):
                    try:
                        val = case.summary_data(cand, time)
                    except Exception:
                        val = None
                    if val is not None:
                        self.pred_data[prim_ind][key] = val
                        break

    def calc_pem(self, time, time_index=None):

        if self.no_flow:
            time_input = time_index
        else:
            time_input = time

        phases = self.pem_input['phases']

        pem_input = {}
        tmp_dyn_var = {}
        # get active porosity
        tmp = self._get_pem_input('PORO')  # self.ecl_case.cell_data('PORO')
        if 'compaction' in self.pem.input:
            multfactor = self._get_pem_input('PORV_RC', time_input)
            pem_input['PORO'] = np.array(multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
        else:
            pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)

        # get active NTG if needed
        if 'ntg' in self.pem.input:
            if self.pem.input['ntg'] == 'no':
                pem_input['NTG'] = None
            else:
                tmp = self._get_pem_input('NTG')
                pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
        else:
            tmp = self._get_pem_input('NTG')
            pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)

        if 'RS' in self.pem.input: #ecl_case.cell_data: # to be more robust!
            tmp = self._get_pem_input('RS', time_input)
            pem_input['RS'] = np.array(tmp[~tmp.mask], dtype=float)
        else:
            pem_input['RS'] = None
            if not hasattr(self, '_rs_warned') or not self._rs_warned:
                print('RS is not a variable in the ecl_case')
                self._rs_warned = True

        # extract pressure
        tmp = self._get_pem_input('PRESSURE', time_input)
        pem_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)

        # convert pressure from Bar to MPa
        if 'press_conv' in self.pem.input and time_input == time:
            pem_input['PRESSURE'] = pem_input['PRESSURE'] * self.pem.input['press_conv']

        if hasattr(self.pem, 'p_init'):
            P_init = self.pem.p_init * np.ones(tmp.shape)[~tmp.mask]
        else:
            P_init = np.array(tmp[~tmp.mask], dtype=float)  # initial pressure is first

        if 'press_conv' in self.pem.input and time_input == time:
            P_init = P_init * self.pem.input['press_conv']

        # extract saturations
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            for var in phases:
                if var in ['WAT', 'GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    pem_input['S{}'.format(var)] = np.clip(pem_input['S{}'.format(var)], 0, 1)

            pem_input['SOIL'] = np.clip(1 - (pem_input['SWAT'] + pem_input['SGAS']), 0, 1)
            saturations = [ np.clip(1 - (pem_input['SWAT'] + pem_input['SGAS']), 0, 1) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                           for ph in phases]
        elif 'WAT' in phases and 'GAS' in phases:  # Smeaheia model using OPM CO2Store
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    pem_input['S{}'.format(var)] = np.clip(pem_input['S{}'.format(var)] , 0, 1)
            pem_input['SWAT'] = 1 - pem_input['SGAS']
            saturations = [1 - (pem_input['SGAS']) if ph == 'WAT' else pem_input['S{}'.format(ph)] for ph in phases]

        elif 'OIL' in phases and 'GAS' in phases:  # Original Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    pem_input['S{}'.format(var)] = np.clip(pem_input['S{}'.format(var)], 0, 1)
            pem_input['SOIL'] = 1 - pem_input['SGAS']
            saturations = [1 - (pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)] for ph in phases]

        else:

            print('Type and number of fluids are unspecified in calc_pem')

        # fluid saturations in dictionary
        # tmp_dyn_var = {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
        for var in phases:
            tmp_dyn_var[f'S{var}'] = pem_input[f'S{var}']

        tmp_dyn_var['PRESSURE'] = pem_input['PRESSURE']
        self.dyn_var.extend([tmp_dyn_var])

        if not self.no_flow:
            keywords = self.ecl_case.arrays(time)
            keywords = [s.strip() for s in keywords]  # Remove leading/trailing spaces
            #for key in self.all_data_types:
            #if 'grav' in key:
            densities = []
            for var in phases:
                # fluid densities
                dens = var + '_DEN'
                if dens in keywords:
                    tmp = self._get_pem_input(dens, time_input)
                    pem_input[dens] = np.array(tmp[~tmp.mask], dtype=float)
                    # extract densities
                    densities.append(pem_input[dens])
                else:
                    densities = None
            # pore volumes at each assimilation step
            if 'RPORV' in keywords:
                tmp = self._get_pem_input('RPORV', time_input)
                pem_input['RPORV'] = np.array(tmp[~tmp.mask], dtype=float)
        else:
            densities = None

        # Get elastic parameters
        if hasattr(self, 'ensemble_member') and (self.ensemble_member is not None) and \
                (self.ensemble_member >= 0):
            self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                dens = densities, ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                ensembleMember=self.ensemble_member)
        else:
            self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                dens = densities, ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init)

    def setup_fwd_run(self, redund_sim):
        super().setup_fwd_run(redund_sim=redund_sim)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i

        # Check if dynamic variables are provided in the state. If that is the case, do not run flow simulator
        if any('sgas' in key for key in state.keys()) or any('swat' in key for key in state.keys()) or any('pressure' in key for key in state.keys()):
            self.state = {}
            for key in state.keys():
                self.state[key] = state[key]
            self.no_flow = True
            #self.pred_data = self.extract_data(member_i)
        #else:
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=del_folder)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if not self.no_flow:
            success = super().call_sim(folder, wait_for_proc)
        else:
            success = True

        if success:
            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')
            phases = self.ecl_case.init.phases
            self.dyn_var = []
            vintage = []
            # loop over seismic vintages
            for v, assim_time in enumerate(self.pem.input['vintage']):
                time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                        dt.timedelta(days=assim_time)

                self.calc_pem(time, v+1)

                # mask the bulk imp. to get proper dimensions
                tmp_value = np.zeros(self.ecl_case.init.shape)
                tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                                   mask=deepcopy(self.ecl_case.init.mask))
                # run filter
                self.pem._filter()
                vintage.append(deepcopy(self.pem.bulkimp))

            if hasattr(self.pem, 'baseline'):  # 4D measurement
                base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                        self.startDate['day']) + dt.timedelta(days=self.pem.baseline)

                self.calc_pem(base_time, 0)

                # mask the bulk imp. to get proper dimensions
                tmp_value = np.zeros(self.ecl_case.init.shape)

                tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                # kill if values are inf or nan
                assert not np.isnan(tmp_value).any()
                assert not np.isinf(tmp_value).any()
                self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                               mask=deepcopy(self.ecl_case.init.mask))
                self.pem._filter()

                # 4D response
                self.pem_result = []
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem - deepcopy(self.pem.bulkimp))
            else:
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem)

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super().extract_data(member)
        self._recover_missing_well_summary_data(member)

        # get the sim2seis from file
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if key in ['bulkimp']:
                    if self.true_prim[1][prim_ind] in self.pem.input['vintage']:
                        v = self.pem.input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.pem_result[v].data.flatten()

class flow_sim2seis(flow):
    """
    Couple the OPM-flow simulator with a sim2seis simulator such that both reservoir quantities and petro-elastic
    quantities can be calculated. Inherit the flow class, and use super to call similar functions.
    """

    def __init__(self, input_dict=None, filename=None, options=None):
        super().__init__(input_dict, filename, options)
        self._getpeminfo(input_dict)

        self.dum_file_root = 'dummy.txt'
        self.dum_entry = str(0)
        self.date_slack = None
        if 'date_slack' in input_dict:
            self.date_slack = int(input_dict['date_slack'])

        # If we want to extract, or evaluate, something uniquely from the ensemble specific run we can
        # run a user defined code to do this.
        self.saveinfo = None
        if 'savesiminfo' in input_dict:
            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(input_dict['savesiminfo'], list):
                self.saveinfo = input_dict['savesiminfo']
            else:
                self.saveinfo = [input_dict['savesiminfo']]

        self.scale = []

    def _getpeminfo(self, input_dict):
        """
        Get, and return, flow and PEM modules
        """
        if 'pem' in input_dict:
            self.pem_input = {}
            for elem in input_dict['pem']:
                if elem[0] == 'model':  # Set the petro-elastic model
                    self.pem_input['model'] = elem[1]
                if elem[0] == 'depth':  # provide the npz of depth values
                    self.pem_input['depth'] = elem[1]
                if elem[0] == 'actnum':  # the npz of actnum values
                    self.pem_input['actnum'] = elem[1]
                if elem[0] == 'baseline':  # the time for the baseline 4D measurement
                    self.pem_input['baseline'] = elem[1]
                if elem[0] == 'vintage':
                    self.pem_input['vintage'] = elem[1]
                    if not type(self.pem_input['vintage']) == list:
                        self.pem_input['vintage'] = [elem[1]]
                if elem[0] == 'ntg':
                    self.pem_input['ntg'] = elem[1]
                if elem[0] == 'press_conv':
                    self.pem_input['press_conv'] = elem[1]
                if elem[0] == 'compaction':
                    self.pem_input['compaction'] = True
                if elem[0] == 'overburden':  # the npz of overburden values
                    self.pem_input['overburden'] = elem[1]
                if elem[0] == 'percentile':  # use for scaling
                    self.pem_input['percentile'] = elem[1]

            pem = getattr(import_module('simulator.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)
            if not hasattr(self.pem, 'input'):
                self.pem.input = self.pem_input

        else:
            self.pem = None

    def setup_fwd_run(self):
        super().setup_fwd_run()

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=True)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        success = super().call_sim(folder, wait_for_proc)

        if success:
            # need an if to check that we have correct sim2seis
            # copy relevant sim2seis files into folder.
            for file in glob.glob('sim2seis_config/*'):
                shutil.copy(file, 'En_' + str(self.ensemble_member) + os.sep)

            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')
            grid = self.ecl_case.grid()

            phases = self.ecl_case.init.phases
            self.dyn_var = []
            vintage = []
            # loop over seismic vintages
            for v, assim_time in enumerate(self.pem.input['vintage']):
                time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                        dt.timedelta(days=assim_time)

                self.calc_pem(time) #mali: update class inherent in flow_rock. Include calc_pem as method in flow_rock

                grdecl.write(f'En_{str(self.ensemble_member)}/Vs{v+1}.grdecl', {
                                 'Vs': self.pem.getShearVel()*.1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                grdecl.write(f'En_{str(self.ensemble_member)}/Vp{v+1}.grdecl', {
                                 'Vp': self.pem.getBulkVel()*.1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                grdecl.write(f'En_{str(self.ensemble_member)}/rho{v+1}.grdecl',
                                 {'rho': self.pem.getDens(), 'DIMENS': grid['DIMENS']}, multi_file=False)

            current_folder = os.getcwd()
            run_folder = current_folder + os.sep + 'En_' + str(self.ensemble_member)
            # The sim2seis is invoked via a shell script. The simulations provides outputs. Run, and get all output. Search
            # for Done. If not finished in reasonable time -> kill
            p = Popen(['./sim2seis.sh', run_folder], stdout=PIPE)
            start = time
            while b'done' not in p.stdout.readline():
                pass

            # Todo: handle sim2seis or pem error

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super().extract_data(member)

        # get the sim2seis from file
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if key in ['sim2seis']:
                    if self.true_prim[1][prim_ind] in self.pem.input['vintage']:
                        result = mat73.loadmat(f'En_{member}/Data_conv.mat')['data_conv']
                        v = self.pem.input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = np.sum(
                            np.abs(result[:, :, :, v]), axis=0).flatten()

class flow_barycenter(flow):
    """
    Couple the OPM-flow simulator with a rock-physics simulator such that both reservoir quantities and petro-elastic
    quantities can be calculated. Inherit the flow class, and use super to call similar functions. In the end, the
    barycenter and moment of interia for the bulkimpedance objects, are returned as observations. The objects are
    identified using k-means clustering, and the number of objects are determined using and elbow strategy.
    """

    def __init__(self, input_dict=None, filename=None, options=None):
        super().__init__(input_dict, filename, options)
        self._getpeminfo(input_dict)

        self.dum_file_root = 'dummy.txt'
        self.dum_entry = str(0)
        self.date_slack = None
        if 'date_slack' in input_dict:
            self.date_slack = int(input_dict['date_slack'])

        # If we want to extract, or evaluate, something uniquely from the ensemble specific run we can
        # run a user defined code to do this.
        self.saveinfo = None
        if 'savesiminfo' in input_dict:
            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(input_dict['savesiminfo'], list):
                self.saveinfo = input_dict['savesiminfo']
            else:
                self.saveinfo = [input_dict['savesiminfo']]

        self.scale = []
        self.pem_result = []
        self.bar_result = []

    def _getpeminfo(self, input_dict):
        """
        Get, and return, flow and PEM modules
        """
        if 'pem' in input_dict:
            self.pem_input = {}
            for elem in input_dict['pem']:
                if elem[0] == 'model':  # Set the petro-elastic model
                    self.pem_input['model'] = elem[1]
                if elem[0] == 'depth':  # provide the npz of depth values
                    self.pem_input['depth'] = elem[1]
                if elem[0] == 'actnum':  # the npz of actnum values
                    self.pem_input['actnum'] = elem[1]
                if elem[0] == 'baseline':  # the time for the baseline 4D measurment
                    self.pem_input['baseline'] = elem[1]
                if elem[0] == 'vintage':
                    self.pem_input['vintage'] = elem[1]
                    if not type(self.pem_input['vintage']) == list:
                        self.pem_input['vintage'] = [elem[1]]
                if elem[0] == 'ntg':
                    self.pem_input['ntg'] = elem[1]
                if elem[0] == 'press_conv':
                    self.pem_input['press_conv'] = elem[1]
                if elem[0] == 'compaction':
                    self.pem_input['compaction'] = True
                if elem[0] == 'overburden':  # the npz of overburden values
                    self.pem_input['overburden'] = elem[1]
                if elem[0] == 'percentile':  # use for scaling
                    self.pem_input['percentile'] = elem[1]
                if elem[0] == 'clusters':  # number of clusters for each barycenter
                    self.pem_input['clusters'] = elem[1]

            pem = getattr(import_module('simulator.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)
            if not hasattr(self.pem, 'input'):
                self.pem.input = self.pem_input

        else:
            self.pem = None

    def setup_fwd_run(self, redund_sim):
        super().setup_fwd_run(redund_sim=redund_sim)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=True)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        success = super().call_sim(folder, wait_for_proc)

        if success:
            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')
            phases = self.ecl_case.init.phases
            #if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            if 'WAT' in phases and 'GAS' in phases:
                vintage = []
                # loop over seismic vintages
                for v, assim_time in enumerate(self.pem.input['vintage']):
                    time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                        dt.timedelta(days=assim_time)
                    pem_input = {}
                    # get active porosity
                    tmp = self.ecl_case.cell_data('PORO')
                    if 'compaction' in self.pem.input:
                        multfactor = self.ecl_case.cell_data('PORV_RC', time)

                        pem_input['PORO'] = np.array(
                            multfactor[~tmp.mask]*tmp[~tmp.mask], dtype=float)
                    else:
                        pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)
                    # get active NTG if needed
                    if 'ntg' in self.pem.input:
                        if self.pem.input['ntg'] == 'no':
                            pem_input['NTG'] = None
                        else:
                            tmp = self.ecl_case.cell_data('NTG')
                            pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
                    else:
                        tmp = self.ecl_case.cell_data('NTG')
                        pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)

                    pem_input['RS'] = None
                    for var in ['SWAT', 'SGAS', 'PRESSURE', 'RS']:
                        try:
                            tmp = self.ecl_case.cell_data(var, time)
                        except:
                            pass
                        # only active, and conv. to float
                        pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)

                    if 'press_conv' in self.pem.input:
                        pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                            self.pem.input['press_conv']

                    tmp = self.ecl_case.cell_data('PRESSURE', 1)
                    if hasattr(self.pem, 'p_init'):
                        P_init = self.pem.p_init*np.ones(tmp.shape)[~tmp.mask]
                    else:
                        # initial pressure is first
                        P_init = np.array(tmp[~tmp.mask], dtype=float)

                    if 'press_conv' in self.pem.input:
                        P_init = P_init*self.pem.input['press_conv']

                    saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                                   for ph in phases]
                    # Get the pressure
                    self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                        ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                        ensembleMember=self.ensemble_member)
                    # mask the bulkimp to get proper dimensions
                    tmp_value = np.zeros(self.ecl_case.init.shape)
                    tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                    self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                                   mask=deepcopy(self.ecl_case.init.mask))
                    # run filter
                    self.pem._filter()
                    vintage.append(deepcopy(self.pem.bulkimp))

            if hasattr(self.pem, 'baseline'):  # 4D measurement
                base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                        self.startDate['day']) + dt.timedelta(days=self.pem.baseline)
                # pem_input = {}
                # get active porosity
                tmp = self.ecl_case.cell_data('PORO')

                if 'compaction' in self.pem.input:
                    multfactor = self.ecl_case.cell_data('PORV_RC', base_time)

                    pem_input['PORO'] = np.array(
                        multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
                else:
                    pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)

                pem_input['RS'] = None
                for var in ['SWAT', 'SGAS', 'PRESSURE', 'RS']:
                    try:
                        tmp = self.ecl_case.cell_data(var, base_time)
                    except:
                        pass
                    # only active, and conv. to float
                    pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)

                if 'press_conv' in self.pem.input:
                    pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                        self.pem.input['press_conv']

                saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                               for ph in phases]
                # Get the pressure
                self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                    ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                    ensembleMember=None)

                # mask the bulkimp to get proper dimensions
                tmp_value = np.zeros(self.ecl_case.init.shape)

                tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                # kill if values are inf or nan
                assert not np.isnan(tmp_value).any()
                assert not np.isinf(tmp_value).any()
                self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                               mask=deepcopy(self.ecl_case.init.mask))
                self.pem._filter()

                # 4D response
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem - deepcopy(self.pem.bulkimp))
            else:
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem)

            #  Extract k-means centers and interias for each element in pem_result
            if 'clusters' in self.pem.input:
                npzfile = np.load(self.pem.input['clusters'], allow_pickle=True)
                n_clusters_list = npzfile['n_clusters_list']
                npzfile.close()
            else:
                n_clusters_list = len(self.pem_result)*[2]
            kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
            for i, bulkimp in enumerate(self.pem_result):
                std = np.std(bulkimp)
                features = np.argwhere(np.squeeze(np.reshape(np.abs(bulkimp), self.ecl_case.init.shape,)) > 3 * std)
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                kmeans = KMeans(n_clusters=n_clusters_list[i], **kmeans_kwargs)
                kmeans.fit(scaled_features)
                kmeans_center = np.squeeze(scaler.inverse_transform(kmeans.cluster_centers_))  # data / measurements
                self.bar_result.append(np.append(kmeans_center, kmeans.inertia_))

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super().extract_data(member)

        # get the barycenters and inertias
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if key in ['barycenter']:
                    if self.true_prim[1][prim_ind] in self.pem.input['vintage']:
                        v = self.pem.input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.bar_result[v].flatten()

class flow_avo(flow_rock, mixIn_multi_data):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        assert 'avo' in input_dict, 'To do AVO simulation, please specify an "AVO" section in the "FWDSIM" part'
        self._get_avo_info()

    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        """
        Setup and run the AVO forward simulator.

        Parameters
        ----------
        state : dict
            Dictionary containing the ensemble state.

        member_i : int
            Index of the ensemble member. any index < 0 (e.g., -1) means the ground truth in synthetic case studies

        del_folder : bool, optional
            Boolean to determine if the ensemble folder should be deleted. Default is False.
        """

        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        #return super().run_fwd_sim(state, member_i, del_folder=del_folder)


        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=del_folder)
        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, run_reservoir_model=None, save_folder=None):
        # replace the sim2seis part (which is unusable) by avo based on Pylops

        if folder is None:
            folder = self.folder
        else:
            self.folder = folder

        if not self.no_flow:
            # call call_sim in flow class (skip flow_rock, go directly to flow which is a parent of flow_rock)
            success = super(flow_rock, self).call_sim(folder, wait_for_proc)
        else:
            success = True

        if success:
            self.get_avo_result(folder, save_folder)

        return success

    @staticmethod
    def _normalize_roi_xy_to_grid(mask_xy, grid_shape_xy):
        """Return ROI mask in the grid-native (nx, ny) orientation."""
        mask_xy = np.asarray(mask_xy).astype(bool)
        nx, ny = grid_shape_xy
        if mask_xy.shape == (nx, ny):
            return mask_xy
        if mask_xy.shape == (ny, nx):
            return mask_xy.T
        raise ValueError(
            f"ROI mask shape {mask_xy.shape} does not match grid shape {(nx, ny)} or {(ny, nx)}."
        )

    @staticmethod
    def _align_grid_xy_to_avo(mask_xy, avo_shape, active_xy_yx=None):
        """Align a grid-native (nx, ny) mask with the actual AVO layout."""
        mask_xy = np.asarray(mask_xy).astype(bool)

        if active_xy_yx is not None:
            if mask_xy.T.shape != active_xy_yx.shape:
                raise ValueError(
                    f"ROI mask/grid shape {mask_xy.shape} is incompatible with active-cell layout {active_xy_yx.shape}."
                )
            roi_active = mask_xy.T[active_xy_yx]
            if len(avo_shape) == 2:
                return np.repeat(roi_active[:, np.newaxis], avo_shape[1], axis=1)
            if len(avo_shape) == 3:
                roi_mask = np.repeat(roi_active[:, np.newaxis], avo_shape[1], axis=1)
                return np.repeat(roi_mask[:, :, np.newaxis], avo_shape[2], axis=2)
            raise ValueError(f"Unsupported active-cell AVO shape {avo_shape}.")

        target_xy = tuple(avo_shape[:2])
        if mask_xy.shape == target_xy:
            aligned_xy = mask_xy
        elif mask_xy.T.shape == target_xy:
            aligned_xy = mask_xy.T
        else:
            raise ValueError(
                f"Could not align ROI mask shape {mask_xy.shape} with AVO xy-shape {target_xy}."
            )

        if len(avo_shape) == 2:
            return aligned_xy
        roi_mask = np.repeat(aligned_xy[:, :, np.newaxis], avo_shape[2], axis=2)
        if len(avo_shape) == 4:
            roi_mask = np.repeat(roi_mask[:, :, :, np.newaxis], avo_shape[3], axis=3)
        return roi_mask

    def get_avo_result(self, folder, save_folder):

        if self.no_flow:
            grid_file = self.pem.input['grid']
            grid = np.load(grid_file)
            zcorn = grid['ZCORN']
            dz = np.diff(zcorn[:, 0, :, 0, :, 0], axis=0)
            # Extract the last layer
            last_layer = dz[-1, :, :]
            # Reshape to ensure it has the same number of dimensions
            last_layer = last_layer.reshape(1, dz.shape[1], dz.shape[2])
            # Concatenate to the original array along the first axis
            dz = np.concatenate([dz, last_layer], axis=0)
            f_dim = [grid['DIMENS'][2], grid['DIMENS'][1], grid['DIMENS'][0]]
        else:
            self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')
            grid = self.ecl_case.grid()
            zcorn = grid['ZCORN']
            ecl_init = ecl.EclipseInit(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')
            dz = ecl_init.cell_data('DZ')
            f_dim = [ecl_init.init.nk, ecl_init.init.nj, ecl_init.init.ni]


        # ecl_init = ecl.EclipseInit(ecl_case)
        # f_dim = [self.ecl_case.init.nk, self.ecl_case.init.nj, self.ecl_case.init.ni]
        #f_dim = [self.NZ, self.NY, self.NX]
        # phases = self.ecl_case.init.phases
        self.dyn_var = []
        # Optional spatial downsampling from AVO config.
        step_x = int(self.avo_config.get('step_x', 1))
        step_y = int(self.avo_config.get('step_y', 1))
        if step_x < 1 or step_y < 1:
            raise ValueError(f"Invalid AVO sampling steps: step_x={step_x}, step_y={step_y}. Both must be >= 1.")

        baseline_mode = str(self.avo_config.get('baseline_mode', 'fixed')).lower()

        def _build_active_mask(actnum, avo_shape):
            """Build an active-trace mask aligned with the first 2 AVO axes."""
            actnum = np.asarray(actnum).astype(bool)
            target_xy = tuple(avo_shape[:2])

            # Active-cell layout: avo_shape[0] is number of active XY traces and
            # avo_shape[1] is time. In this case data are already restricted to
            # active traces, so active mask is all True.
            if actnum.ndim == 3:
                active_xy_yx = np.any(actnum, axis=0)  # (ny, nx) for ACTNUM=(nz,ny,nx)
                n_active_xy = int(np.sum(active_xy_yx))
                if len(avo_shape) >= 2 and avo_shape[0] == n_active_xy and avo_shape[1] != actnum.shape[1]:
                    return np.ones(avo_shape, dtype=bool)

            active_xy = None
            for ax in range(actnum.ndim):
                reduced = np.any(actnum, axis=ax)
                if reduced.shape == target_xy:
                    active_xy = reduced
                    break
                if reduced.ndim == 2 and reduced.T.shape == target_xy:
                    active_xy = reduced.T
                    break

            if active_xy is None:
                raise ValueError(
                    f"Could not align ACTNUM shape {actnum.shape} with AVO xy-shape {target_xy}."
                )

            active_mask = np.repeat(active_xy[:, :, np.newaxis], avo_shape[2], axis=2)
            if len(avo_shape) == 4:
                active_mask = np.repeat(active_mask[:, :, :, np.newaxis], avo_shape[3], axis=3)
            return active_mask

        def _get_grid_coordinates(grid):
             """Extract grid corner coordinates and compute cell centers."""
             coord = grid['COORD']

             # Assuming COORD shape is (nx+1, ny+1, ..., ...) for ECLIPSE format
             nx = coord.shape[0] - 1
             ny = coord.shape[1] - 1

             # Compute cell center coordinates by averaging corner coordinates.
             x_centers = np.zeros((nx, ny))
             y_centers = np.zeros((nx, ny))

             for i in range(nx):
                 for j in range(ny):
                     x_centers[i, j] = 0.25 * (
                         coord[i, j, 0, 0] + coord[i + 1, j, 0, 0] +
                         coord[i, j + 1, 0, 0] + coord[i + 1, j + 1, 0, 0]
                     )
                     y_centers[i, j] = 0.25 * (
                         coord[i, j, 0, 1] + coord[i + 1, j, 0, 1] +
                         coord[i, j + 1, 0, 1] + coord[i + 1, j + 1, 0, 1]
                     )

             return x_centers, y_centers

        def _load_completion_ij(data_file):
             """Read COMPDAT completion I/J pairs (0-based) from a DATA deck."""
             if not os.path.exists(data_file):
                 return []
             out = []
             section = None
             with open(data_file, 'r', errors='ignore') as fh:
                 for raw in fh:
                     line = raw.strip()
                     if not line or line.startswith('--'):
                         continue
                     up = line.upper()
                     if up.startswith('COMPDAT'):
                         section = 'COMPDAT'
                         continue
                     if line == '/':
                         section = None
                         continue
                     if section != 'COMPDAT':
                         continue
                     toks = line.replace('/', ' ').strip().split()
                     if len(toks) < 3:
                         continue
                     try:
                         out.append((int(toks[1]) - 1, int(toks[2]) - 1))
                     except ValueError:
                         continue
             return out
        
        def _build_roi_mask(avo_cube):
             """Build optional ROI mask for assimilation.

             Modes (avo.roi_mode):
               - "none": keep all traces
               - "file": use roi_mask_file only
               - "ellipse" / "rectangle": connected ROI around well completions + strong AVO signal
             """
             avo_shape = np.shape(avo_cube)
             coord = grid['COORD']
             actnum = grid['ACTNUM']
             
             # Get grid dimensions and coordinates
             nx = coord.shape[0] - 1
             ny = coord.shape[1] - 1
             x_grid, y_grid = _get_grid_coordinates(grid)

             # Detect whether grids are laid out as [row=y, col=x].
             def _med_abs_diff(a, axis):
                 d = np.diff(a, axis=axis)
                 d = d[np.isfinite(d)]
                 return float(np.median(np.abs(d))) if d.size else 0.0

             vx0 = _med_abs_diff(x_grid, 0)
             vx1 = _med_abs_diff(x_grid, 1)
             vy0 = _med_abs_diff(y_grid, 0)
             vy1 = _med_abs_diff(y_grid, 1)
             use_ji = (vx1 + vy0) > (vx0 + vy1)
             
             # Detect active-cell layout and recover grid-space NX/NY from ACTNUM.
             active_cell_layout = False
             active_xy_yx = None
             if np.asarray(grid['ACTNUM']).ndim == 3:
                 actnum_bool = np.asarray(grid['ACTNUM']).astype(bool)
                 active_xy_yx = np.any(actnum_bool, axis=0)  # (ny, nx)
                 n_active_xy = int(np.sum(active_xy_yx))
                 if len(avo_shape) >= 2 and avo_shape[0] == n_active_xy and avo_shape[1] != actnum.shape[1]:
                     active_cell_layout = True

             roi_mode = str(self.avo_config.get('roi_mode', 'ellipse')).lower()
             margin_m = float(self.avo_config.get('roi_margin_m', 500.0))
             signal_pct = float(self.avo_config.get('roi_signal_percentile', 85.0))
             use_signal = bool(self.avo_config.get('roi_use_signal', False))

             # Default: keep all traces if ROI is disabled.
             roi_xy = np.ones((nx, ny), dtype=bool)

             if roi_mode not in ('none', 'off', 'all'):
                 # Build seed points from well completions.
                 data_path = folder + os.sep + self.file + '.DATA' if folder[-1] != os.sep else folder + self.file + '.DATA'
                 completion_ij = _load_completion_ij(data_path)
                 seed_x = []
                 seed_y = []
                 for i_idx, j_idx in completion_ij:
                     if use_ji:
                         if 0 <= j_idx < nx and 0 <= i_idx < ny:
                             seed_x.append(float(x_grid[j_idx, i_idx]))
                             seed_y.append(float(y_grid[j_idx, i_idx]))
                     else:
                         if 0 <= i_idx < nx and 0 <= j_idx < ny:
                             seed_x.append(float(x_grid[i_idx, j_idx]))
                             seed_y.append(float(y_grid[i_idx, j_idx]))

                 # Add strong AVO-signal points so ROI follows saturation-change-sensitive area.
                 if use_signal and np.ndim(avo_cube) >= 3:
                     dyn = np.nanpercentile(np.abs(avo_cube), 95.0, axis=tuple(range(2, np.ndim(avo_cube))))
                     dyn = np.asarray(dyn, dtype=float)
                     if dyn.shape == (nx, ny) and np.any(np.isfinite(dyn)):
                         thr = np.nanpercentile(dyn[np.isfinite(dyn)], signal_pct)
                         ii, jj = np.where(dyn >= thr)
                         for a, b in zip(ii, jj):
                             seed_x.append(float(x_grid[a, b]))
                             seed_y.append(float(y_grid[a, b]))

                 # Fallback to active grid envelope if seeds are sparse.
                 if len(seed_x) < 2:
                     actnum_bool = np.asarray(actnum).astype(bool)
                     active_indices = np.where(np.any(actnum_bool, axis=0))
                     if len(active_indices[0]) > 0:
                         seed_x.extend(list(np.asarray(x_grid[active_indices], dtype=float).reshape(-1)))
                         seed_y.extend(list(np.asarray(y_grid[active_indices], dtype=float).reshape(-1)))

                 x_range = self.avo_config.get('x_range', None)
                 y_range = self.avo_config.get('y_range', None)
                 if x_range is not None and isinstance(x_range, list) and len(x_range) == 2:
                     x_min = float(min(x_range[0], x_range[1]))
                     x_max = float(max(x_range[0], x_range[1]))
                 elif seed_x:
                     x_min = float(np.min(seed_x))
                     x_max = float(np.max(seed_x))
                 else:
                     x_min = float(np.min(x_grid))
                     x_max = float(np.max(x_grid))

                 if y_range is not None and isinstance(y_range, list) and len(y_range) == 2:
                     y_min = float(min(y_range[0], y_range[1]))
                     y_max = float(max(y_range[0], y_range[1]))
                 elif seed_y:
                     y_min = float(np.min(seed_y))
                     y_max = float(np.max(seed_y))
                 else:
                     y_min = float(np.min(y_grid))
                     y_max = float(np.max(y_grid))

                 if roi_mode == 'rectangle':
                     roi_xy = (
                         (x_grid >= (x_min - margin_m)) & (x_grid <= (x_max + margin_m)) &
                         (y_grid >= (y_min - margin_m)) & (y_grid <= (y_max + margin_m))
                     )
                 elif roi_mode in ('ellipse', 'auto'):
                     x_center = 0.5 * (x_min + x_max)
                     y_center = 0.5 * (y_min + y_max)
                     x_radius = max(1.0, 0.5 * (x_max - x_min) + margin_m)
                     y_radius = max(1.0, 0.5 * (y_max - y_min) + margin_m)
                     dx = (x_grid - x_center) / x_radius
                     dy = (y_grid - y_center) / y_radius
                     roi_xy = (dx * dx + dy * dy) <= 1.0

             # Optional mask file overrides mask if provided.
             roi_mask_file = self.avo_config.get('roi_mask_file', None)
             if roi_mode == 'file' and roi_mask_file is not None:
                 with np.load(roi_mask_file, allow_pickle=True) as f:
                     key = 'mask' if 'mask' in f.files else f.files[0]
                     roi_xy = self._normalize_roi_xy_to_grid(f[key], (nx, ny))

             roi_mask = self._align_grid_xy_to_avo(
                 roi_xy,
                 avo_shape,
                 active_xy_yx=active_xy_yx if active_cell_layout else None,
             )

             # Save coordinate system for plotting (always available from grid)
             self.x_grid_centers = x_grid
             self.y_grid_centers = y_grid

             return roi_mask

        def _save_wavelet_mask(mask, vintage_idx):
             """Save compress-ready 3D mask that matches current AVO vectorization."""
             mask = np.asarray(mask, dtype=bool)
             if mask.ndim == 4:
                 nx, ny, nt, n_ang = mask.shape
                 mask_3d = mask.reshape(nx, ny, nt * n_ang)
             elif mask.ndim == 3:
                 mask_3d = mask
             else:
                 return
             np.savez(f'mask_{vintage_idx}_3d.npz', mask=mask_3d)
        if 'baseline' in self.pem_input:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.pem_input['baseline'])


            self.calc_pem(base_time,0)
            # vp, vs, density in reservoir
            self.calc_velocities(folder, save_folder, grid, -1, f_dim)

            if not self.no_flow:
                # vp, vs, density in reservoir
                vp, vs, rho = self.calc_velocities(folder, save_folder, grid, 0, f_dim)

                # avo data
                # self._calc_avo_props()
                avo_data_baseline, Rpp_baseline, vp_baseline, vs_baseline, _ = self._calc_avo_props_active_cells(grid, vp, vs, rho, dz, zcorn)
                if avo_data_baseline.ndim >= 3 and avo_data_baseline.shape[0] == self.NY and avo_data_baseline.shape[1] == self.NX:
                    kept_data = avo_data_baseline[::step_x, ::step_y, :].copy()
                    avo_data_baseline[:] = np.nan
                    avo_data_baseline[::step_x, ::step_y, :] = kept_data
                    baseline_sample_mask = np.zeros(np.shape(avo_data_baseline), dtype=bool)
                    baseline_sample_mask[::step_x, ::step_y, :] = True
                else:
                    baseline_sample_mask = np.ones(np.shape(avo_data_baseline), dtype=bool)
                baseline_active_mask = _build_active_mask(grid['ACTNUM'], avo_data_baseline.shape)
                baseline_roi_mask = _build_roi_mask(avo_data_baseline)
                baseline_structural_mask = np.logical_and(
                    np.logical_and(baseline_active_mask, baseline_sample_mask),
                    baseline_roi_mask
                )
                if not np.any(baseline_structural_mask):
                    raise ValueError("Baseline AVO mask is empty; check ACTNUM orientation and AVO geometry.")
                #rho_baseline = rho_sample
                tmp = self._get_pem_input('PRESSURE', base_time)
                PRESSURE_baseline = np.array(tmp[~tmp.mask], dtype=float)
                tmp = self._get_pem_input('SGAS', base_time)
                SGAS_baseline = np.array(tmp[~tmp.mask], dtype=float)

                # Rolling baseline starts at configured baseline and advances per vintage.
                rolling_ref_avo_data = avo_data_baseline
                rolling_ref_Rpp = Rpp_baseline
                rolling_ref_vs = vs_baseline
                rolling_ref_vp = vp_baseline
                rolling_ref_PRESSURE = PRESSURE_baseline
                rolling_ref_SGAS = SGAS_baseline
                rolling_ref_structural_mask = baseline_structural_mask

            else:
                file_name = f"avo_vint0_{folder}.npz" if folder[-1] != os.sep \
                    else f"avo_vint0_{folder[:-1]}.npz"

                avo_baseline = np.load(file_name, allow_pickle=True)['avo_bl']
                #Rpp_baseline = np.load(file_name, allow_pickle=True)['Rpp_bl']
                #vs_baseline = np.load(file_name, allow_pickle=True)['Vs_bl']
                #vp_baseline = np.load(file_name, allow_pickle=True)['Vp_bl']
                #rho_baseline = np.load(file_name, allow_pickle=True)['Rho_bl']

        vintage = []
        # loop over seismic vintages
        for v, assim_time in enumerate(self.pem_input['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                   dt.timedelta(days=assim_time)

            # extract dynamic variables from simulation run
            self.calc_pem(time, v+1)

            # vp, vs, density in reservoir
            vp, vs, rho = self.calc_velocities(folder, save_folder, grid, v+1, f_dim)

            # avo data
            #self._calc_avo_props()
            avo_data, Rpp, vp_sample, vs_sample, _ = self._calc_avo_props_active_cells(grid, vp, vs, rho, dz, zcorn)
            #avo_data = avo_data[::step_x, ::step_y, :]
            # make mask for wavelet decomposition
            if avo_data.ndim >= 3 and avo_data.shape[0] == self.NY and avo_data.shape[1] == self.NX:
                kept_data = avo_data[::step_x, ::step_y, :].copy()
                avo_data[:] = np.nan
                avo_data[::step_x, ::step_y, :] = kept_data
                sample_mask = np.zeros(np.shape(avo_data), dtype=bool)
                sample_mask[::step_x, ::step_y, :] = True
            else:
                sample_mask = np.ones(np.shape(avo_data), dtype=bool)
            active_mask = _build_active_mask(grid['ACTNUM'], avo_data.shape)
            roi_mask = _build_roi_mask(avo_data)
            mask = np.logical_and(
                np.logical_and(active_mask, sample_mask),
                roi_mask
            )
            if not np.any(mask):
                raise ValueError("AVO mask is empty; check ACTNUM orientation and AVO geometry.")
            np.savez(f'mask_{v}.npz', mask=mask)
            _save_wavelet_mask(mask, v)
            avo = np.where(np.isfinite(avo_data), avo_data, 0.0)[mask]
            if avo.size == 0:
                raise ValueError("AVO data selection is empty after masking.")

            tmp = self._get_pem_input('PRESSURE', time)
            PRESSURE = np.array(tmp[~tmp.mask], dtype=float)
            tmp = self._get_pem_input('SGAS', time)
            SGAS = np.array(tmp[~tmp.mask], dtype=float)


            # MLIE: implement 4D avo
            if 'baseline' in self.pem_input:  # 4D measurement
                if baseline_mode == 'rolling':
                    ref_avo_data = rolling_ref_avo_data
                    ref_Rpp = rolling_ref_Rpp
                    ref_vs = rolling_ref_vs
                    ref_vp = rolling_ref_vp
                    ref_PRESSURE = rolling_ref_PRESSURE
                    ref_SGAS = rolling_ref_SGAS
                    ref_structural_mask = rolling_ref_structural_mask
                else:
                    ref_avo_data = avo_data_baseline
                    ref_Rpp = Rpp_baseline
                    ref_vs = vs_baseline
                    ref_vp = vp_baseline
                    ref_PRESSURE = PRESSURE_baseline
                    ref_SGAS = SGAS_baseline
                    ref_structural_mask = baseline_structural_mask

                if avo_data.shape != ref_avo_data.shape:
                    raise ValueError(
                        f"Baseline/current AVO cube shapes differ: {avo_data.shape} vs {ref_avo_data.shape}."
                    )
                delta_avo = avo_data - ref_avo_data
                delta_avo = np.where(np.isfinite(delta_avo), delta_avo, 0.0)
                valid_mask = np.logical_and(mask, ref_structural_mask)
                if not np.any(valid_mask):
                    raise ValueError("4D AVO valid mask is empty after intersecting baseline and vintage masks.")
                avo = delta_avo[valid_mask]
                if avo.size == 0:
                    raise ValueError("4D AVO selection is empty after baseline subtraction and masking.")
                Rpp = Rpp - ref_Rpp
                vs_sample = vs_sample - ref_vs
                vp_sample = vp_sample - ref_vp
                PRESSURE = PRESSURE - ref_PRESSURE
                SGAS = SGAS - ref_SGAS

                if baseline_mode == 'rolling':
                    rolling_ref_avo_data = np.copy(avo_data)
                    rolling_ref_Rpp = np.copy(Rpp + ref_Rpp)
                    rolling_ref_vs = np.copy(vs_sample + ref_vs)
                    rolling_ref_vp = np.copy(vp_sample + ref_vp)
                    rolling_ref_PRESSURE = np.copy(PRESSURE + ref_PRESSURE)
                    rolling_ref_SGAS = np.copy(SGAS + ref_SGAS)
                    rolling_ref_structural_mask = np.copy(mask)

            
            # XLUO: self.ensemble_member < 0 => reference reservoir model in synthetic case studies
            # the corresonding (noisy) data are observations in data assimilation
            if 'add_synthetic_noise' in self.input_dict and self.ensemble_member < 0:
                non_nan_idx = np.argwhere(~np.isnan(avo))
                data_std = np.std(avo[non_nan_idx])
                if self.input_dict['add_synthetic_noise'][0] == 'snr':
                    noise_std = np.sqrt(self.input_dict['add_synthetic_noise'][1]) * data_std
                    avo[non_nan_idx] += noise_std * np.random.randn(avo[non_nan_idx].size, 1)
            else:
                noise_std = 0.0  # simulated data don't contain noise

            vintage.append(deepcopy(avo))

            # Add spatial coordinates to save dict if available
            coord_data = {}
            if hasattr(self, 'x_grid_centers') and hasattr(self, 'y_grid_centers'):
                coord_data = {'x_grid': self.x_grid_centers, 'y_grid': self.y_grid_centers}
            
            if v == 0:
                save_dic = {'avo': avo, 'noise_std': noise_std, 'Rpp': Rpp, 'Vs': vs_sample, 'Vp': vp_sample, 'PRESSURE': PRESSURE, 'SGAS': SGAS, 'base_time': base_time, **self.avo_config, **self.pem_input, **coord_data}
                #save_dic = {'avo': avo, 'noise_std': noise_std, 'Rpp': Rpp, 'Vs': vs_sample, 'Vp': vp_sample, 'rho': rho_sample, #**self.avo_config,
                #        'Vs_bl': vs_baseline, 'Vp_bl': vp_baseline, 'avo_bl': avo_baseline, 'Rpp_bl': Rpp_baseline, 'rho_bl': rho_baseline, **self.avo_config}
                #save_dic = {'avo': avo, 'noise_std': noise_std, 'Rpp': Rpp, 'Vs': vs_sample, 'Vp': vp_sample,
                #            **self.avo_config}
            else:
                save_dic = {'avo': avo, 'noise_std': noise_std, 'Rpp': Rpp, 'Vs': vs_sample, 'Vp': vp_sample, 'PRESSURE': PRESSURE, 'SGAS': SGAS, **coord_data}#, 'Rpp': Rpp, 'Vs': vs_sample, 'Vp': vp_sample,
                            #'rho': rho_sample,  **self.avo_config}

            if save_folder is not None:
                file_name = save_folder + os.sep + f"avo_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"avo_vint{v}.npz"
                #np.savez(file_name, **save_dic)
            else:
                file_name = folder + os.sep + f"avo_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"avo_vint{v}.npz"
                #file_name_rec = 'Ensemble_results/' + f"avo_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                #    else 'Ensemble_results/' + f"avo_vint{v}_{folder[:-1]}.npz"
                #np.savez(file_name_rec, **save_dic)
                # with open(file_name, "wb") as f:
                #    dump(**save_dic, f)
            np.savez(file_name, **save_dic)
        # 4D response
        self.avo_result = []
        for i, elem in enumerate(vintage):
            self.avo_result.append(elem)

    def calc_velocities(self, folder, save_folder, grid, v, f_dim):
        # The properties in pem are only given in the active cells
        # indices of active cells:

        true_indices = np.where(grid['ACTNUM'])

        vp = np.full(f_dim, np.nan)
        vp[true_indices] = (self.pem.getBulkVel())
        vs = np.full(f_dim, np.nan)
        vs[true_indices] = (self.pem.getShearVel())
        rho = np.full(f_dim, np.nan)
        rho[true_indices] = (self.pem.getDens())



        ## Debug
        #bulkmod = np.full(f_dim, np.nan)
        #bulkmod[true_indices] = self.pem.getBulkMod()
        #self.shearmod = np.full(f_dim, np.nan)
        #self.shearmod[true_indices] = self.pem.getShearMod()
        #self.poverburden = np.full(f_dim, np.nan)
        #self.poverburden[true_indices] = self.pem.getOverburdenP()
        #self.pressure = np.full(f_dim, np.nan)
        #self.pressure[true_indices] = self.pem.getPressure()
        #self.peff = np.full(f_dim, np.nan)
        #self.peff[true_indices] = self.pem.getPeff()
        #porosity = np.full(f_dim, np.nan)
        #porosity[true_indices] = self.pem.getPorosity()
        #if self.dyn_var:
           # sgas = np.full(f_dim, np.nan)
           # sgas[true_indices] = self.dyn_var[v]['SGAS']
           #  soil = np.full(f_dim, np.nan)
           #  soil[true_indices] = self.dyn_var[v]['SOIL']
           # pdyn = np.full(f_dim, np.nan)
           # pdyn[true_indices] = self.dyn_var[v]['PRESSURE']
        #
        #if self.dyn_var is None:
        #    save_dic = {'vp': vp, 'vs': vs, 'rho': rho}#, 'bulkmod': self.bulkmod, 'shearmod': self.shearmod,
        #            #'Pov': self.poverburden, 'P': self.pressure,  'Peff': self.peff, 'por': porosity} # for debugging
        #else:
        #    save_dic = {'vp': vp, 'vs': vs, 'rho': rho}#, 'por': porosity, 'sgas': sgas, 'Pd': pdyn}

        #if save_folder is not None:
        #    file_name = save_folder + os.sep + f"vp_vs_rho_vint{v}.npz" if save_folder[-1] != os.sep \
        #        else save_folder + f"vp_vs_rho_vint{v}.npz"
        #    np.savez(file_name, **save_dic)
        #else:
        #    file_name_rec = 'Ensemble_results/' + f"vp_vs_rho_vint{v}_{folder}.npz" if folder[-1] != os.sep \
        #            else 'Ensemble_results/' + f"vp_vs_rho_vint{v}_{folder[:-1]}.npz"
        #    np.savez(file_name_rec, **save_dic)
        # for debugging
        return vp, vs, rho

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super(flow_rock, self).extract_data(member)

        # get the avo from file
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'avo' in key:
                    if self.true_prim[1][prim_ind] in self.pem.input['vintage']:
                        idx = self.pem.input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.avo_result[idx].flatten()

    def _get_avo_info(self, avo_config=None):
         """
         AVO configuration
         """
         # list of configuration parameters in the "AVO" section
         config_para_list = ['dz', 'tops', 'angle', 'frequency', 'wave_len', 'vp_shale', 'vs_shale',
                             'den_shale', 't_min', 't_max', 't_sampling', 'pp_func',
                             'step_x', 'step_y', 'x_range', 'y_range', 'roi_mask_file', 'baseline_mode',
                             'use_ellipse', 'roi_mode', 'roi_margin_m', 'roi_signal_percentile', 'roi_use_signal']
         if 'avo' in self.input_dict:
             self.avo_config = {}
             for elem in self.input_dict['avo']:
                 assert elem[0] in config_para_list, f'Property {elem[0]} not supported'
                 if elem[0] == 'vintage' and not isinstance(elem[1], list):
                     elem[1] = [elem[1]]
                 self.avo_config[elem[0]] = elem[1]

             # if only one angle is considered, convert self.avo_config['angle'] into a list, as required later
             if isinstance(self.avo_config['angle'], float):
                 self.avo_config['angle'] = [self.avo_config['angle']]

             # Optional AVO spatial sampling strides for x/y (default no thinning).
             self.avo_config['step_x'] = int(self.avo_config.get('step_x', 1))
             self.avo_config['step_y'] = int(self.avo_config.get('step_y', 1))
             if self.avo_config['step_x'] < 1 or self.avo_config['step_y'] < 1:
                 raise ValueError(
                     f"Invalid AVO config: step_x={self.avo_config['step_x']}, "
                     f"step_y={self.avo_config['step_y']}. Both must be >= 1."
                 )

             # Optional baseline mode for 4D AVO: 'fixed' (legacy) or 'rolling'.
             self.avo_config['baseline_mode'] = str(self.avo_config.get('baseline_mode', 'fixed')).lower()
             if self.avo_config['baseline_mode'] not in ('fixed', 'rolling'):
                 raise ValueError(
                     f"Invalid AVO config: baseline_mode={self.avo_config['baseline_mode']}. "
                     "Expected 'fixed' or 'rolling'."
                 )
             
             # ROI selection mode used by assimilation mask builder.
             if 'roi_mode' in self.avo_config:
                 self.avo_config['roi_mode'] = str(self.avo_config['roi_mode']).lower()
             else:
                 # Backward compatibility with legacy boolean switch.
                 use_ellipse = bool(self.avo_config.get('use_ellipse', True))
                 self.avo_config['roi_mode'] = 'ellipse' if use_ellipse else 'none'

             if self.avo_config['roi_mode'] not in ('none', 'off', 'all', 'ellipse', 'rectangle', 'file', 'auto'):
                 raise ValueError(
                     f"Invalid AVO config: roi_mode={self.avo_config['roi_mode']}. "
                     "Expected one of 'ellipse', 'rectangle', 'file', 'none'."
                 )

             self.avo_config['roi_margin_m'] = float(self.avo_config.get('roi_margin_m', 500.0))
             if self.avo_config['roi_margin_m'] < 0:
                 raise ValueError(
                     f"Invalid AVO config: roi_margin_m={self.avo_config['roi_margin_m']}. Must be >= 0."
                 )

             self.avo_config['roi_signal_percentile'] = float(self.avo_config.get('roi_signal_percentile', 85.0))
             if not (0.0 <= self.avo_config['roi_signal_percentile'] <= 100.0):
                 raise ValueError(
                     f"Invalid AVO config: roi_signal_percentile={self.avo_config['roi_signal_percentile']}. "
                     "Must be in [0, 100]."
                 )

             # self._get_DZ(file=self.avo_config['dz'])  # =>self.DZ
             kw_file = {'DZ': self.avo_config['dz'], 'TOPS': self.avo_config['tops']}
             self._get_props(kw_file)
             self.overburden = self.pem_input['overburden']

             # make sure that the "pylops" package is installed
             # See https://github.com/PyLops/pylops
             self.pp_func = getattr(import_module('pylops.avo.avo'), self.avo_config['pp_func'])

         else:
             self.avo_config = None

    def _get_props(self, kw_file):
        # extract properties (specified by keywords) in (possibly) different files
        # kw_file: a dictionary contains "keyword: file" pairs
        # Note that all properties are reshaped into the reservoir model dimension (NX, NY, NZ)
        # using the "F" order
        for kw in kw_file:
            file = kw_file[kw]
            if file.endswith('.npz'):
                with np.load(file) as f:
                    exec(f'self.{kw} = f[ "{kw}" ]')
                    self.NX, self.NY, self.NZ = f['NX'], f['NY'], f['NZ']
            else:
                try:
                    self.NX = int(self.input_dict['dimension'][0])
                    self.NY = int(self.input_dict['dimension'][1])
                    self.NZ = int(self.input_dict['dimension'][2])
                except:
                    for item in self.input_dict['pem']:
                        if item[0] == 'dimension':
                            dimension = item[1]
                            break
                    if len(dimension)== 2:
                        self.NX = int(dimension[0])
                        self.NY = int(1)
                        self.NZ = int(dimension[1])
                    else:
                        self.NX = int(dimension[0])
                        self.NY = int(dimension[1])
                        self.NZ = int(dimension[2])
            #    reader = GRDECL_Parser(filename=file)
            #    reader.read_GRDECL()
            #    exec(f"self.{kw} = reader.{kw}.reshape((reader.NX, reader.NY, reader.NZ), order='F')")
            #    self.NX, self.NY, self.NZ = reader.NX, reader.NY, reader.NZ
            #    eval(f'np.savez("./{kw}.npz", {kw}=self.{kw}, NX=self.NX, NY=self.NY, NZ=self.NZ)')

    def _calc_avo_props(self, dt=0.0005):
        # dt is the fine resolution sampling rate
        # convert properties in reservoir model to time domain
        vp_shale = self.avo_config['vp_shale']  # scalar value (code may not work for matrix value)
        vs_shale = self.avo_config['vs_shale']  # scalar value
        rho_shale = self.avo_config['den_shale']  # scalar value

        # Two-way travel time of the top of the reservoir
        # TOPS[:, :, 0] corresponds to the depth profile of the reservoir top on the first layer
        top_res = 2 * self.TOPS[:, :, 0] / vp_shale

        # Cumulative traveling time through the reservoir in vertical direction
        cum_time_res = np.cumsum(2 * self.DZ / self.vp, axis=0) + top_res[:, :, np.newaxis]

        # assumes under burden to be constant. No reflections from under burden. Hence set travel time to under burden very large
        underburden = top_res + np.nanmax(cum_time_res)

        # total travel time
        # cum_time = np.concatenate((top_res[:, :, np.newaxis], cum_time_res), axis=2)
        cum_time = np.concatenate((top_res[np.newaxis, :, :], cum_time_res, underburden[np.newaxis, :, :]), axis=0)


        # add overburden and underburden of Vp, Vs and Density
        vp = np.concatenate((vp_shale * np.ones((1, self.NY, self.NX)),
                             self.vp, vp_shale * np.ones((1, self.NY, self.NX))), axis=0)
        vs = np.concatenate((vs_shale * np.ones((1, self.NY, self.NX)),
                             self.vs, vs_shale * np.ones((1, self.NY, self.NX))), axis=0)
        rho = np.concatenate((rho_shale * np.ones((1, self.NY, self.NX)),
                              self.rho, rho_shale * np.ones((1, self.NY, self.NX))), axis=0)

        # search for the lowest grid cell thickness and sample the time according to
        # that grid thickness to preserve the thin layer effect
        time_sample = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], dt)
        if time_sample.shape[0] == 1:
            time_sample = time_sample.reshape(-1)
        time_sample = np.tile(time_sample, (self.NX, self.NY, 1))

        vp_sample = np.tile(vp[:, :, 1][..., np.newaxis], (1, 1, time_sample.shape[2]))
        vs_sample = np.tile(vs[:, :, 1][..., np.newaxis], (1, 1, time_sample.shape[2]))
        rho_sample = np.tile(rho[:, :, 1][..., np.newaxis], (1, 1, time_sample.shape[2]))

        for m in range(self.NX):
            for l in range(self.NY):
                for k in range(time_sample.shape[2]):
                    # find the right interval of time_sample[m, l, k] belonging to, and use
                    # this information to allocate vp, vs, rho
                    idx = np.searchsorted(cum_time[m, l, :], time_sample[m, l, k], side='left')
                    idx = idx if idx < len(cum_time[m, l, :]) else len(cum_time[m, l, :]) - 1
                    vp_sample[m, l, k] = vp[m, l, idx]
                    vs_sample[m, l, k] = vs[m, l, idx]
                    rho_sample[m, l, k] = rho[m, l, idx]

        # Ricker wavelet
        wavelet, t_axis, wav_center = ricker(np.arange(0, self.avo_config['wave_len'], dt),
                                             f0=self.avo_config['frequency'])


        # Travel time corresponds to reflectivity series
        t = time_sample[:, :, 0:-1]

        # interpolation time
        t_interp = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], self.avo_config['t_sampling'])
        trace_interp = np.zeros((self.NX, self.NY, len(t_interp)))

        # number of pp reflection coefficients in the vertical direction

        nz_rpp = vp_sample.shape[2] - 1

        for i in range(len(self.avo_config['angle'])):
            angle = self.avo_config['angle'][i]
            Rpp = self.pp_func(vp_sample[:, :, :-1], vs_sample[:, :, :-1], rho_sample[:, :, :-1],
                           vp_sample[:, :, 1:], vs_sample[:, :, 1:], rho_sample[:, :, 1:], angle)

            for m in range(self.NX):
                for l in range(self.NY):
                    # convolution with the Ricker wavelet
                    conv_op = Convolve1D(nz_rpp, h=wavelet, offset=wav_center, dtype="float32")
                    w_trace = conv_op * Rpp[m, l, :]

                    # Sample the trace into regular time interval
                    f = interp1d(np.squeeze(t[m, l, :]), np.squeeze(w_trace),
                                kind='nearest', fill_value='extrapolate')
                    trace_interp[m, l, :] = f(t_interp)

            if i == 0:
                avo_data = trace_interp  # 3D
            elif i == 1:
                avo_data = np.stack((avo_data, trace_interp), axis=-1)  # 4D
            else:
                avo_data = np.concatenate((avo_data, trace_interp[:, :, :, np.newaxis]), axis=3)  # 4D

        self.avo_data = avo_data

    def _calc_avo_props_active_cells(self, grid, vp, vs, rho, dz, zcorn, dt=0.0005):
        # dt is the fine resolution sampling rate
        # convert properties in reservoir model to time domain
        vp_shale = self.avo_config['vp_shale']  # scalar value (code may not work for matrix value)
        vs_shale = self.avo_config['vs_shale']  # scalar value
        rho_shale = self.avo_config['den_shale']  # scalar value


        actnum = grid['ACTNUM']
        # Find indices where the boolean array is True
        active_indices = np.where(actnum)
        c, a, b = active_indices

        # Two-way travel time tp the top of the reservoir
        top_res = 2 * zcorn[0, 0, :, 0, :, 0] / vp_shale

        # depth difference between cells in z-direction:
        depth_differences = dz#(zcorn[:, 0, :, 0, :, 0] , axis=0)


        # Cumulative traveling time through the reservoir in vertical direction
         #cum_time_res = 2 * zcorn[:, 0, :, 0, :, 0] / self.vp  + top_res[np.newaxis, :, :]
        cum_time_res = np.cumsum(2 * depth_differences / vp, axis=0) + top_res[np.newaxis, :, :]
        # assumes under burden to be constant. No reflections from under burden. Hence set travel time to under burden very large
        underburden = top_res + np.nanmax(cum_time_res)

        # total travel time
        # cum_time = np.concat enate((top_res[:, :, np.newaxis], cum_time_res), axis=2)
        cum_time = np.concatenate((top_res[np.newaxis, :, :], cum_time_res, underburden[np.newaxis, :, :]), axis=0)

        # add overburden and underburden values for  Vp, Vs and Density
        vp = np.concatenate((vp_shale * np.ones((1, self.NY, self.NX)),
                             vp, vp_shale * np.ones((1, self.NY, self.NX))), axis=0)
        vs = np.concatenate((vs_shale * np.ones((1, self.NY, self.NX)),
                             vs, vs_shale * np.ones((1, self.NY, self.NX))), axis=0)
        rho = np.concatenate((rho_shale * np.ones((1, self.NY, self.NX)),
                              rho, rho_shale * np.ones((1, self.NY, self.NX))), axis=0)


        # Combine a and b into a 2D array (each column represents a vector)
        ab = np.column_stack((a, b))

        # Extract unique rows and get the indices of those unique rows
        unique_rows, indices = np.unique(ab, axis=0, return_index=True)

        # search for the lowest grid cell thickness and sample the time according to
        # that grid thickness to preserve the thin layer effect
        time_sample = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], dt)
        if time_sample.shape[0] == 1:
            time_sample = time_sample.reshape(-1)
        time_sample = np.tile(time_sample, (len(indices), 1))

        vp_sample = np.zeros((len(indices), time_sample.shape[1]))
        vs_sample = np.zeros((len(indices), time_sample.shape[1]))
        rho_sample = np.zeros((len(indices), time_sample.shape[1]))

        for ind in range(len(indices)):
            for k in range(time_sample.shape[1]):
                # find the right interval of time_sample[m, l, k] belonging to, and use
                # this information to allocate vp, vs, rho
                idx = np.searchsorted(cum_time[:, a[indices[ind]], b[indices[ind]]], time_sample[ind, k], side='left')
                idx = idx if idx < len(cum_time[:, a[indices[ind]], b[indices[ind]]]) else len(
                    cum_time[:,a[indices[ind]], b[indices[ind]]]) - 1
                vp_sample[ind, k] = vp[idx, a[indices[ind]], b[indices[ind]]]
                vs_sample[ind, k] = vs[idx, a[indices[ind]], b[indices[ind]]]
                rho_sample[ind, k] = rho[idx, a[indices[ind]], b[indices[ind]]]

        # Ricker wavelet
        wavelet, t_axis, wav_center = ricker(np.arange(0, self.avo_config['wave_len']-dt, dt),
                                             f0=self.avo_config['frequency'])

        # Travel time corresponds to reflectivity series
        t = time_sample[:, 0:-1]

        # interpolation time
        t_interp = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], self.avo_config['t_sampling'])
        trace_interp = np.zeros((len(indices), len(t_interp)))

        # number of pp reflection coefficients in the vertical direction
        nz_rpp = vp_sample.shape[1] - 1
        conv_op = Convolve1D(nz_rpp, h=wavelet, offset=wav_center, dtype="float32")

        avo_data = []
        Rpp = []
        for i in range(len(self.avo_config['angle'])):
            angle = self.avo_config['angle'][i]
            angle = np.atleast_1d(angle)
            Rpp = self.pp_func(vp_sample[:, :-1], vs_sample[:, :-1], rho_sample[:, :-1],
                               vp_sample[:, 1:], vs_sample[:, 1:], rho_sample[:, 1:], angle)

            for ind in range(len(indices)):
                # convolution with the Ricker wavelet

                w_trace = conv_op * Rpp[ind, :]

                # Sample the trace into regular time interval
                f = interp1d(np.squeeze(t[ind, :]), np.squeeze(w_trace),
                             kind='nearest', fill_value='extrapolate')
                trace_interp[ind, :] = f(t_interp)

            if i == 0:
                avo_data = trace_interp  # 3D
            elif i == 1:
                avo_data = np.stack((avo_data, trace_interp), axis=-1)  # 4D
            else:
                avo_data = np.concatenate((avo_data, trace_interp[:, :, :, np.newaxis]), axis=3)  # 4D

        # Reshape avo_data from trace format (n_traces, time, [...angles]) back to spatial grid format (ny, nx, time, [...angles])
        # unique_rows contains (y, x) indices for each trace
        ny = self.NY
        nx = self.NX

        if avo_data.ndim == 2:
            # Single angle: (n_traces, time) -> (ny, nx, time)
            avo_data_spatial = np.full((ny, nx, avo_data.shape[1]), np.nan)
            for ind in range(len(unique_rows)):
                y, x = unique_rows[ind]
                avo_data_spatial[y, x, :] = avo_data[ind, :]
            avo_data = avo_data_spatial
        elif avo_data.ndim == 3:
            # Multiple angles: (n_traces, time, n_angles) -> (ny, nx, time, n_angles)
            n_angles = avo_data.shape[2]
            avo_data_spatial = np.full((ny, nx, avo_data.shape[1], n_angles), np.nan)
            for ind in range(len(unique_rows)):
                y, x = unique_rows[ind]
                avo_data_spatial[y, x, :, :] = avo_data[ind, :, :]
            avo_data = avo_data_spatial

        return avo_data, Rpp, vp_sample, vs_sample, rho_sample


    @classmethod
    def _reformat3D_then_flatten(cls, array, flatten=True, order="F"):
        """
        XILU: Quantities read by "EclipseData.cell_data" are put in the axis order of [nz, ny, nx]. To be consisent with
        ECLIPSE/OPM custom, we need to change the axis order. We further flatten the array according to the specified order
        """
        array = np.array(array)
        if len(array.shape) != 1:  # if array is a 1D array, then do nothing
            assert isinstance(array, np.ndarray) and len(array.shape) == 3, "Only 3D numpy array are supported"

            # axis [0 (nz), 1 (ny), 2 (nx)] -> [2 (nx), 1 (ny), 0 (nz)]
            new_array = np.transpose(array, axes=[2, 1, 0])
            if flatten:
                new_array = new_array.flatten(order=order)

            return new_array
        else:
            return array

class flow_grav(flow_rock, mixIn_multi_data):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        self.grav_input = {}
        assert 'grav' in input_dict, 'To do GRAV simulation, please specify an "GRAV" section in the "FWDSIM" part'
        self._get_grav_info()

    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        #return super().run_fwd_sim(state, member_i, del_folder=del_folder)
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder)
        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder
        else:
            self.folder = folder

        # run flow  simulator
        # success = True
        success = super(flow_rock, self).call_sim(folder, True)

        # use output from flow simulator to forward model gravity response
        if success:
            self.get_grav_result(folder, save_folder)

        return success

    def get_grav_result(self, folder, save_folder):
        if self.no_flow:
            grid_file = self.pem.input['grid']
            grid = np.load(grid_file)
        else:
            self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')
            grid = self.ecl_case.grid()


        self.dyn_var = []

        # cell centers
        #cell_centre = self.find_cell_centre(grid)

        # receiver locations
        # Make a mesh of the area
        pad = self.grav_config.get('padding', 1500)  # 3 km padding around the reservoir
        if 'padding' not in self.grav_config:
            print('Please specify extent of measurement locations (Padding in input file), using 1.5 km as default')
        dxy = self.grav_config.get('meas_spacing', 1500)  #
        if 'meas_spacing' not in self.grav_config:
            print('Please specify measurement spacing in input file, using 1.5 km as default')
        if 'seabed' in self.grav_config and isinstance(self.grav_config['seabed'], str) is True:
            file_path = self.grav_config['seabed']
            water_depth = self.get_seabed_depths(file_path)
        else:
            water_depth = self.grav_config.get('water_depth', 300)
            if 'water_depth' not in self.grav_config:
                print('Please specify water depths in input file, using 300 m as default')
        well_coord = self.grav_config.get('well_coord', None)

        pos = self.measurement_locations(grid, water_depth, pad, dxy, well_coord)

        # loop over vintages with gravity acquisitions
        grav_struct = {}
        baseline_mode = str(self.grav_config.get('baseline_mode', 'fixed')).lower()
        if baseline_mode not in ('fixed', 'rolling'):
            raise ValueError(
                f"Invalid GRAV config: baseline_mode={baseline_mode}. Expected 'fixed' or 'rolling'."
            )

        if 'baseline' in self.grav_config:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.grav_config['baseline'])
            # porosity, saturation, densities, and fluid mass at time of baseline survey
            grav_base, cutoff_base = self.calc_mass(base_time, 0, cutoff_base=None)
            rolling_base = deepcopy(grav_base)


        else:
            # seafloor gravity only works in 4D mode
            grav_base = None
            cutoff_base = None
            print('Need to specify Baseline survey for gravity in input file')

        for v, assim_time in enumerate(self.grav_config['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                   dt.timedelta(days=assim_time)

            # porosity, saturation, densities, and fluid mass at individual time-steps
            grav_struct[v], _ = self.calc_mass(time, v+1, cutoff_base=cutoff_base)  # calculate the mass of each fluid in each grid cell



        vintage = []


        for v, assim_time in enumerate(self.grav_config['vintage']):
            base_ref = rolling_base if baseline_mode == 'rolling' else grav_base
            dg = self.calc_grav(grid, base_ref, grav_struct[v], pos)
            vintage.append(deepcopy(dg))

            #save_dic = {'grav': dg, **self.grav_config}
            save_dic = {
                'grav': dg, 'P_vint': grav_struct[v]['PRESSURE'], 'rho_gas_vint':grav_struct[v]['GAS_DEN'],
                'meas_location': pos, **self.grav_config,
                **{key: grav_struct[v][key] - base_ref[key] for key in grav_struct[v].keys()}
            }
            if save_folder is not None:
                file_name = save_folder + os.sep + f"grav_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"grav_vint{v}.npz"
            else:
                file_name = folder + os.sep + f"grav_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"grav_vint{v}.npz"
                prior_folder =  'Prior_ensemble_results'
                try:
                    files = os.listdir(prior_folder)
                    filename_to_check = f"grav_vint{v}_{folder}.npz"

                    if filename_to_check in files:
                        file_name_rec = 'Ensemble_results/' + f"grav_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                            else 'Ensemble_results/' + f"grav_vint{v}_{folder[:-1]}.npz"
                    else:
                        file_name_rec = 'Prior_ensemble_results/' + f"grav_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                            else 'Prior_ensemble_results/' + f"grav_vint{v}_{folder[:-1]}.npz"

                except:
                    file_name_rec = 'Ensemble_results/' + f"grav_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                        else 'Ensemble_results/' + f"grav_vint{v}_{folder[:-1]}.npz"
                #np.savez(file_name_rec, **save_dic)

            np.savez(file_name, **save_dic)

            if baseline_mode == 'rolling':
                rolling_base = deepcopy(grav_struct[v])


        # 4D response
        self.grav_result = []
        for i, elem in enumerate(vintage):
            self.grav_result.append(elem)

    def calc_mass(self, time, time_index = None, cutoff_base = None):

        if self.no_flow:
            time_input = time_index
        else:
            time_input = time

        # fluid phases given as input; accept both "WAT GAS" and ["WAT", "GAS"]
        phases_raw = self.pem.input.get('phases', '')
        phases = []
        if isinstance(phases_raw, str):
            phases = phases_raw.upper().split()
        elif isinstance(phases_raw, (list, tuple, set, np.ndarray)):
            for item in phases_raw:
                phases.extend(str(item).upper().split())
        else:
            phases = str(phases_raw).upper().split()
        #
        grav_input = {}
        tmp_dyn_var = {}


        tmp = self._get_pem_input('RPORV', time_input)

        tmp, cutoff = self.filter_rporv(tmp, cutoff = cutoff_base, method='percentile', percentile=90)
        grav_input['RPORV'] = np.array(tmp[~tmp.mask], dtype=float)

        tmp = self._get_pem_input('PRESSURE', time_input)
        #if time_input == time_index and time_index > 0: # to be activiated in case on inverts for Delta Pressure
        #    # Inverts for changes in dynamic variables using time-lapse data
        #    tmp_baseline = self._get_pem_input('PRESSURE', 0)
        #    tmp = tmp + tmp_baseline
        grav_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)
        # convert pressure from Bar to MPa
        if 'press_conv' in self.pem.input and time_input == time:
            grav_input['PRESSURE'] = grav_input['PRESSURE'] * self.pem.input['press_conv']
        #else:
        #    print('Keyword RPORV missing from simulation output, need updated pore volumes at each assimilation step')
        # extract saturation
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            for var in phases:
                if var in ['WAT', 'GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    #if time_input == time_index and time_index > 0: # to be activated in case on inverts for Delta S
                    #    # Inverts for changes in dynamic variables using time-lapse data
                    #    tmp_baseline = self._get_pem_input('S{}'.format(var), 0)
                    #    tmp = tmp + tmp_baseline
                    #tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] > 1] = 1
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] < 0] = 0

            grav_input['SOIL'] = 1 - (grav_input['SWAT'] + grav_input['SGAS'])
            grav_input['SOIL'][grav_input['SOIL'] > 1] = 1
            grav_input['SOIL'][grav_input['SOIL'] < 0] = 0


            tmp_dyn_var['SWAT'] = grav_input['SWAT']  # = {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
            tmp_dyn_var['SGAS'] = grav_input['SGAS']
            tmp_dyn_var['SOIL'] = grav_input['SOIL']


        elif 'WAT' in phases and 'GAS' in phases:  # Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    #if time_input == time_index and time_index > 0: # to be activated in case on inverts for Delta S
                        # Inverts for changes in dynamic variables using time-lapse data
                    #    tmp_baseline = self._get_pem_input('S{}'.format(var), 0)
                    #    tmp = tmp + tmp_baseline
                    #tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] > 1] = 1
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] < 0] = 0

            grav_input['SWAT'] = 1 - (grav_input['SGAS'])

            # fluid saturation
            tmp_dyn_var['SWAT'] =  grav_input['SWAT'] #= {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
            tmp_dyn_var['SGAS'] = grav_input['SGAS']

        elif 'OIL' in phases and 'GAS' in phases:  # Original Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    #if time_input == time_index and time_index > 0: # to be activated in case on inverts for Delta S
                        # Inverts for changes in dynamic variables using time-lapse data
                    #    tmp_baseline = self._get_pem_input('S{}'.format(var), 0)
                    #    tmp = tmp + tmp_baseline
                    #tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] > 1] = 1
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] < 0] = 0

            grav_input['SOIL'] = 1 - (grav_input['SGAS'])

            # fluid saturation
            tmp_dyn_var['SOIL'] = grav_input['SOIL'] #= {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
            tmp_dyn_var['SGAS'] = grav_input['SGAS']

        else:
            print('Type and number of fluids are unspecified in calc_mass')


        # fluid densities
        for var in phases:
            dens = var + '_DEN'
            #tmp = self.ecl_case.cell_data(dens, time)
            if self.no_flow:
                if  any('pressure' in key for key in self.state.keys()):
                    if 'press_conv' in self.pem.input:
                        conv2pa = 1e6 #MPa to Pa
                    else:
                        conv2pa = 1e5  # Bar to Pa

                    if var == 'GAS':
                        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:
                            tmp = PropsSI('D', 'T', 298.15, 'P', grav_input['PRESSURE']*conv2pa, 'Methane')
                        elif 'WAT' in phases and 'GAS' in grav_input['PRESSURE']:  # Smeaheia model T = 37 C
                            tmp = PropsSI('D', 'T', 310.15, 'P', grav_input['PRESSURE']*conv2pa, 'CO2')
                        mask = np.zeros(tmp.shape, dtype=bool)
                        tmp = np.ma.array(data=tmp, dtype=tmp.dtype, mask=mask)
                    elif var == 'WAT':
                        tmp = PropsSI('D', 'T|liquid', 298.15, 'P', grav_input['PRESSURE']*conv2pa, 'Water')
                        mask = np.zeros(tmp.shape, dtype=bool)
                        tmp = np.ma.array(data=tmp, dtype=tmp.dtype, mask=mask)
                    else:
                        tmp = self._get_pem_input(dens, time_input)
                else:
                    tmp = self._get_pem_input(dens, time_input)
                grav_input[dens] = np.array(tmp[~tmp.mask], dtype=float)
                tmp_dyn_var[dens] = grav_input[dens]
            else:
                tmp = self._get_pem_input(dens, time_input)
                grav_input[dens] = np.array(tmp[~tmp.mask], dtype=float)
                tmp_dyn_var[dens] = grav_input[dens]


        tmp_dyn_var['PRESSURE'] = grav_input['PRESSURE']
        tmp_dyn_var['RPORV'] = grav_input['RPORV']
        self.dyn_var.extend([tmp_dyn_var])

            #fluid masses
        for var in phases:
            mass = var + '_mass'
            grav_input[mass] = grav_input[var + '_DEN'] * grav_input['S' + var] * grav_input['RPORV']

        return grav_input, cutoff

    def calc_grav(self, grid, grav_base, grav_repeat, pos):

        cell_centre = self.find_cell_centre(grid)
        x = cell_centre[0]
        y = cell_centre[1]
        z = cell_centre[2]

        # Initialize dg as a zero array, with shape depending on the condition
        # assumes the length of each vector gives the total number of measurement points
        n_meas = (len(pos['x']))
        dg = np.zeros(n_meas)  # 1D array for dg
        dg[:] = np.nan

        # fluid phases given as input; accept both "WAT GAS" and ["WAT", "GAS"]
        phases_raw = self.pem.input.get('phases', '')
        phases = []
        if isinstance(phases_raw, str):
            phases = phases_raw.upper().split()
        elif isinstance(phases_raw, (list, tuple, set, np.ndarray)):
            for item in phases_raw:
                phases.extend(str(item).upper().split())
        else:
            phases = str(phases_raw).upper().split()
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:
            dm  = grav_repeat['OIL_mass'] + grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['OIL_mass'] + grav_base['WAT_mass'] + grav_base['GAS_mass'])

        elif 'OIL' in phases and 'GAS' in phases:  # Original Smeaheia model
            dm = grav_repeat['OIL_mass'] + grav_repeat['GAS_mass'] - (grav_base['OIL_mass'] + grav_base['GAS_mass'])
            # dm = grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['WAT_mass'] + grav_base['GAS_mass'])

        elif 'WAT' in phases and 'GAS' in phases:  # Smeaheia or SPE11 model
            dm  = grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['WAT_mass'] + grav_base['GAS_mass'])
            #dm = grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['WAT_mass'] + grav_base['GAS_mass'])

        else:
            dm = None
            print('Type and number of fluids are unspecified in calc_grav')


        for j in range(n_meas):

            # Calculate dg for the current measurement location (j, i)
            dg_tmp = (z - pos['z'][j]) / ((x - pos['x'][j]) ** 2 + (y - pos['y'][j]) ** 2 + (
                                z - pos['z'][j]) ** 2) ** (3 / 2)

            dg[j] = np.dot(dg_tmp, dm)
            #print(f'Progress: {j + 1}/{n_meas}')  # Mimicking wait bar

        # Scale dg by the constant
        dg *= 6.67e-3

        return dg

    def _get_grav_info(self, grav_config=None):
        """
        GRAV configuration
        """
        # list of configuration parameters in the "Grav" section of teh pipt file
        config_para_list = ['baseline', 'vintage', 'method', 'model', 'poisson', 'compressibility',
                            'z_base', 'meas_spacing', 'padding', 'seabed', 'water_depth', 'well_coord', 'baseline_mode']

        if 'grav' in self.input_dict:
            self.grav_config = {}
            for elem in self.input_dict['grav']:
                assert elem[0] in config_para_list, f'Property {elem[0]} not supported'
                if elem[0] == 'vintage' and not isinstance(elem[1], list):
                    elem[1] = [elem[1]]
                self.grav_config[elem[0]] = elem[1]
            self.grav_config['baseline_mode'] = str(self.grav_config.get('baseline_mode', 'fixed')).lower()
            if self.grav_config['baseline_mode'] not in ('fixed', 'rolling'):
                raise ValueError(
                    f"Invalid GRAV config: baseline_mode={self.grav_config['baseline_mode']}. "
                    "Expected 'fixed' or 'rolling'."
                )
        else:
            self.grav_config = None

    def extract_data(self, member):
        # start by getting the data from the flow simulator i.e. prod. and inj. data
        super(flow_rock, self).extract_data(member)
        self._recover_missing_well_summary_data(member)

        # get the gravity data from results
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'grav' in key:
                    if self.true_prim[1][prim_ind] in self.grav_config['vintage']:
                        v = self.grav_config['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.grav_result[v].flatten()

class flow_seafloor_disp(flow_rock, mixIn_multi_data):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        assert 'sea_disp' in input_dict, 'To do subsidence/uplift simulation, please specify an "SEA_DISP" section in the pipt file'
        self._get_disp_info()


    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder

        # run flow  simulator
        # success = True
        success = super(flow_rock, self).call_sim(folder, True)

        # use output from flow simulator to forward model gravity response
        if success:
            # calculate gravity data based on flow simulation output
            self.get_displacement_result(folder, save_folder)


        return success

    def get_displacement_result(self, folder, save_folder):
        if self.no_flow:
            grid_file = self.pem.input['grid']
            grid = np.load(grid_file)
        else:
            self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')
            grid = self.ecl_case.grid()

        self.dyn_var = []

        # receiver locations
        pad = self.disp_config.get('padding', 1500)  # 3 km padding around the reservoir
        if 'padding' not in self.disp_config:
            print('Please specify extent of measurement locations, padding in input file, using 1.5 km as default')
        dxy = self.disp_config.get('meas_spacing', 1500)  #
        if 'meas_spacing' not in self.disp_config:
            print('Please specify grid spacing in input file, using 1.5 km as default')
        if 'seabed' in self.disp_config and isinstance(self.disp_config['seabed'], str) is True:
            file_path = self.disp_config['seabed']
            water_depth = self.get_seabed_depths(file_path)
        else:
            water_depth = self.disp_config.get('water_depth', 300)
            if 'water_depth' not in self.disp_config:
                print('Please specify water depths in input file, using 300 m as default')
        pos = self.measurement_locations(grid, water_depth, pad, dxy)

        # loop over vintages with gravity acquisitions
        disp_struct = {}
        baseline_mode = str(self.disp_config.get('baseline_mode', 'fixed')).lower()
        if baseline_mode not in ('fixed', 'rolling'):
            raise ValueError(
                f"Invalid SEA_DISP config: baseline_mode={baseline_mode}. Expected 'fixed' or 'rolling'."
            )

        if 'baseline' in self.disp_config:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.disp_config['baseline'])
            # pore volume at time of baseline survey
            disp_base, cutoff_base  = self.get_pore_volume(base_time, 0)
            rolling_base = deepcopy(disp_base)

        else:
            # seafloor  displacement only work in 4D mode
            disp_base = None
            cutoff_base = None
            print('Need to specify Baseline survey for displacement modelling in input file')

        for v, assim_time in enumerate(self.disp_config['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                   dt.timedelta(days=assim_time)

            # pore volume and pressure at individual time-steps
            disp_struct[v], _ = self.get_pore_volume(time, v+1)  # calculate the mass of each fluid in each grid cell

        vintage = []

        for v, assim_time in enumerate(self.disp_config['vintage']):
            # calculate subsidence and uplift
            base_ref = rolling_base if baseline_mode == 'rolling' else disp_base
            dz_seafloor = self.map_z_response(base_ref, disp_struct[v], grid, pos)
            vintage.append(deepcopy(dz_seafloor))

            save_dic = {'sea_disp': dz_seafloor, 'meas_location': pos, **self.disp_config}
            if save_folder is not None:
                file_name = save_folder + os.sep + f"sea_disp_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"sea_disp_vint{v}.npz"
            else:
                file_name = folder + os.sep + f"sea_disp_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"sea_disp_vint{v}.npz"
                file_name_rec = 'Ensemble_results/' + f"sea_disp_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                    else 'Ensemble_results/' + f"sea_disp_vint{v}_{folder[:-1]}.npz"
                np.savez(file_name_rec, **save_dic)

            np.savez(file_name, **save_dic)

            if baseline_mode == 'rolling':
                rolling_base = deepcopy(disp_struct[v])


        # 4D response
        self.disp_result = []
        for i, elem in enumerate(vintage):
            self.disp_result.append(elem)

    def get_pore_volume(self, time, time_index = None, cutoff_base = None):

        if self.no_flow:
            time_input = time_index
        else:
            time_input = time

        #
        disp_input = {}
        tmp_dyn_var = {}


        tmp = self._get_pem_input('RPORV', time_input)
        tmp, cutoff = self.filter_rporv(tmp, cutoff=cutoff_base, method='percentile', percentile=90)
        disp_input['RPORV'] = np.array(tmp[~tmp.mask], dtype=float)

        tmp = self._get_pem_input('PRESSURE', time_input)
        #if time_input == time_index and time_index > 0: # to be activiated in case on inverts for Delta Pressure
        #    # Inverts for changes in dynamic variables using time-lapse data
        #    tmp_baseline = self._get_pem_input('PRESSURE', 0)
        #    tmp = tmp + tmp_baseline
        disp_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)
        # convert pressure from Bar to MPa
        if 'press_conv' in self.pem.input and time_input == time:
            disp_input['PRESSURE'] = disp_input['PRESSURE'] * self.pem.input['press_conv']
        #else:
        #    print('Keyword RPORV missing from simulation output, need pdated pore volumes at each assimilation step')


        tmp_dyn_var['PRESSURE'] = disp_input['PRESSURE']
        tmp_dyn_var['RPORV'] = disp_input['RPORV']
        self.dyn_var.extend([tmp_dyn_var])

        return disp_input, cutoff

    def compute_horizontal_distance(self, pos, x, y):
        dx = pos['x'][:, np.newaxis] - x
        dy = pos['y'][:, np.newaxis] - y
        rho = np.sqrt(dx ** 2 + dy ** 2).flatten()
        return rho

    def _build_disp_kernel(self, grid, pos, poisson, compressibility, z_base, model):
        """Precompute displacement Green's-function matrix for one grid/receiver geometry."""
        # coordinates of active cell centres
        cell_centre = self.find_cell_centre(grid)
        x = np.asarray(cell_centre[0], dtype=float)
        y = np.asarray(cell_centre[1], dtype=float)
        z = np.asarray(cell_centre[2], dtype=float)

        # receiver coordinates
        rx = np.asarray(pos['x'], dtype=float)
        ry = np.asarray(pos['y'], dtype=float)
        rz = np.asarray(pos['z'], dtype=float)

        # elastic modulus in the same units as used in the original formulation
        E = ((1 + poisson) * (1 - 2 * poisson)) / ((1 - poisson) * compressibility)

        if model == 'van_Opstal':
            component = ["Geertsma_vertical", "System_3_vertical"]
        else:
            component = ["Geertsma_vertical"]

        geertsma_mode = str(self.disp_config.get('geertsma_mode', 'integral')).lower()

        # Early receiver-distance truncation (same criterion as in compute_deformation_transfer).
        trunc_factor = float(self.disp_config.get('receiver_trunc_factor', 3.0))
        if trunc_factor <= 0:
            raise ValueError(
                f"Invalid SEA_DISP config: receiver_trunc_factor={trunc_factor}. Must be > 0."
            )

        # Precompute horizontal distances from all receivers to all active cells.
        dx = rx[:, np.newaxis] - x[np.newaxis, :]
        dy = ry[:, np.newaxis] - y[np.newaxis, :]
        rho_mat = np.sqrt(dx * dx + dy * dy)

        n_rec = rx.size
        n_cells = x.size
        kernel = np.zeros((n_rec, n_cells), dtype=float)

        for j in range(n_cells):
            # Skip receivers that are guaranteed to be zero according to the far-field cutoff.
            max_rho = np.abs(z[j] - rz) * trunc_factor
            active = rho_mat[:, j] <= max_rho
            if not np.any(active):
                continue

            # Optional fast Geertsma mode: closed-form kernel approximation (much faster).
            if model != 'van_Opstal' and geertsma_mode == 'fast':
                rho_km = rho_mat[active, j] / 1e3
                dz_km = np.abs(z[j] - rz[active]) / 1e3
                denom = np.power(rho_km * rho_km + dz_km * dz_km, 1.5)
                col = np.zeros_like(rho_km, dtype=float)
                nz = denom > 0
                # Same closed-form Geertsma transfer used elsewhere in this file.
                col[nz] = (2.0 * dz_km[nz] / denom[nz]) * 1e-6
                kernel[active, j] = col
                continue

            # dV=1 gives a linear kernel column; runtime response is kernel @ dV.
            thh, trb = self.compute_deformation_transfer(
                rz[active], z[j], z_base, rho_mat[active, j], poisson, E, 1.0, component
            )
            kernel[active, j] = (thh + trb) if model == 'van_Opstal' else thh

        return kernel

    def map_z_response(self, base, repeat, grid, pos):
        """
        Maps out subsidence and uplift based either on the simulation
        model pressure drop (method = 'pressure') or simulated change in pore volume
        using either the van Opstal or Geertsma forward model

        Arguments:
        base -- A dictionary containing baseline  pressures and pore volumes.
        repeat -- A dictionary containing pressures and pore volumes at repeat measurements.

        compute subsidence at position 'pos', b

        Output is modeled subsidence in cm.

        """

        # Method to compute pore volume change
        method = self.disp_config['method'].lower()

        # Forward model to compute subsidence/uplift response
        model = self.disp_config['model'].lower()

        if self.disp_config['poisson'] > 0.5:
            poisson = 0.5
            print('Poisson\'s ratio exceeds physical limits, setting it to 0.5')
        else:
            poisson = self.disp_config['poisson']

        # Depth of rigid basement
        z_base = self.disp_config['z_base']

        compressibility = self.disp_config['compressibility'] # 1/MPa

        # compute pore volume change between baseline and repeat survey
        # based on the reservoir pore volumes in the individual vintages
        if method == 'pressure':
            dV = base['RPORV'] * (base['PRESSURE'] - repeat['PRESSURE']) * compressibility
        else:
            dV = base['RPORV'] - repeat['RPORV']

        # Build (or reuse) linear response kernel for this geometry/configuration.
        cache_key = (
            model,
            str(self.disp_config.get('geertsma_mode', 'integral')).lower(),
            float(poisson),
            float(compressibility),
            float(z_base),
            float(self.disp_config.get('receiver_trunc_factor', 3.0)),
            int(np.asarray(grid['ACTNUM']).sum()),
            int(np.asarray(pos['x']).size),
        )
        cached = getattr(self, '_disp_kernel_cache', None)
        if cached is None or cached.get('key') != cache_key:
            kernel = self._build_disp_kernel(grid, pos, poisson, compressibility, z_base, model)
            self._disp_kernel_cache = {'key': cache_key, 'kernel': kernel}
        else:
            kernel = cached['kernel']

        dz = np.asarray(kernel @ np.asarray(dV, dtype=float), dtype=float)

        # Convert from meters to centimeters
        dz *= 100

        return dz

    def compute_van_opstal_transfer_function(self, z_res, z_base, rho, poisson):
        """
        Compute the Van Opstal transfer function.

        Args:
        z_res -- Numpy array of depths to reservoir cells [m].
        z_base -- Distance to the basement [m].
        rho -- Numpy array of horizontal distances in the field [m].
        poisson -- Poisson's ratio.

        Returns:
        T -- Numpy array of the transfer function values.
        T_geertsma -- Numpy array of the Geertsma transfer function values.
        """

        # Change to km scale
        rho = rho / 1e3
        z_res = z_res / 1e3
        z_base = z_base / 1e3


        # Find lambda max (to optimize Hilbert transform)
        cutoff = 1e-10  # Function value at max lambda
        try:
            lambda_max = fsolve(lambda x: 4 * (2 * x * z_base + 1) / (3 - 4 * poisson) * np.exp(
                x * (np.max(z_res) - 2 * z_base)) - cutoff, 10)[0]
        except:
            lambda_max = 10  # Default value if unable to solve for max lambda

        lambda_vals = np.linspace(0, lambda_max, 100)
        # range of lateral distances between measurement location and reservoir cells
        nj = len(rho)
        # range of vertical distances between measurement location and reservoir cells
        ni = len(z_res)
        # initialize
        t_van_opstal = np.zeros((ni, nj))

        # input function to make a hankel transform of order 0 of
        c_t = self.van_opstal(lambda_vals, z_res[0], z_base, poisson)

        h_t, i_t = self.h_t(c_t, lambda_vals, rho)  # Extract integrand
        t_van_opstal[0, :] = (2 * z_res[0] / (rho ** 2 + z_res[0] ** 2) ** (3 / 2)) + h_t / (2 * np.pi)

        for i in range(1, ni):
            C = self.van_opstal(lambda_vals, z_res[i], z_base, poisson)
            h_t = self.h_t(C, lambda_vals, rho, i_t)
            t_van_opstal[i, :] = (2 * z_res[i] / (rho ** 2 + z_res[i] ** 2) ** (3 / 2)) + h_t / (2 * np.pi)

        t_van_opstal *= 1e-6  # Convert back to meters

        t_geertsma = (2 * z_res[:, np.newaxis] / ((np.ones((ni, 1)) * rho) ** 2 + (z_res[:, np.newaxis]) ** 2) ** (
                    3 / 2)) * 1e-6

        return t_van_opstal, t_geertsma

    def van_opstal(self, lambda_vals, z_res, z_base, poisson):
        """
        Compute the Van Opstal transfer function.

        Args:
        lambda_vals -- Numpy array of lambda values.
        z_res -- Depth to reservoir [m].
        z_base -- Distance to the basement [m].
        poisson -- Poisson's ratio.

        Returns:
        value -- Numpy array of computed values.
        """

        term1 = np.exp(lambda_vals * z_res) * (2 * lambda_vals * z_base + 1)
        term2 = np.exp(-lambda_vals * z_res) * (
                4 * lambda_vals ** 2 * z_base ** 2 + 2 * lambda_vals * z_base + (3 - 4 * poisson) ** 2)

        term3_numer = (3 - 4 * poisson) * (
                np.exp(-lambda_vals * (2 * z_base + z_res)) - np.exp(-lambda_vals * (2 * z_base - z_res)))
        term3_denom = 2 * ((1 - 2 * poisson) ** 2 + lambda_vals ** 2 * z_base ** 2 + (3 - 4 * poisson) * np.cosh(
            lambda_vals * z_base) ** 2)

        value = term1 - term2 - (term3_numer / term3_denom)

        return value

    def hankel_transform_order_0(self, f, r_max, num_points=1000):
        """
        Computes the Hankel transform of order 0 of a function f(r).

        Parameters:
        - f: callable, the function to transform, f(r)
        - r_max: float, upper limit of the integral (approximate infinity)
        - num_points: int, number of points for numerical integration

        Returns:
        - k_values: array of k values
        - H_k: array of Hankel transform evaluated at k_values
        """
        r = np.linspace(0, r_max, num_points)
        dr = r[1] - r[0]
        f_r = f(r)

        def integrand(r, k):
            return f(r) * j0(k * r) * r

        # Define a range of k values to evaluate
        k_min, k_max = 0, 10  # adjust as needed
        k_values = np.linspace(k_min, k_max, 100)

        H_k = []

        for k in k_values:
            # Perform numerical integration over r
            result, _ = quad(integrand, 0, r_max, args=(k,))
            H_k.append(result)

        return k_values, np.array(H_k)

    def makeL(self, poisson, k, c, A_g, eps, lambda_):
        L = A_g * (
                (4 * poisson - 3 + 2 * k * lambda_) * np.exp(-lambda_ * (k + c))
                - np.exp(lambda_ * eps * (k - c))
        )
        return L

    def makeM(self, poisson, k, c, A_g, eps, lambda_):
        M = A_g * (
                (4 * poisson - 3 - 2 * k * lambda_) * np.exp(-lambda_ * (k + c))
                - eps * np.exp(lambda_ * eps * (k - c))
        )
        return M

    def makeDelta(self, poisson, k, lambda_):
        Delta = (
                (4 * poisson - 3) * np.cosh(k * lambda_) ** 2
                - (k * lambda_) ** 2
                - (1 - 2 * poisson) ** 2
        )
        return Delta

    def makeB(self, poisson, k, c, A_g, eps, lambda_):
        L = self.makeL(poisson, k, c, A_g, eps, lambda_)
        M = self.makeM(poisson, k, c, A_g, eps, lambda_)
        Delta = self.makeDelta(poisson, k, lambda_)

        numerator = (
                lambda_ * L * (2 * (1 - poisson) * np.cosh(k * lambda_) - lambda_ * k * np.sinh(k * lambda_))
                + lambda_ * M * ((1 - 2 * poisson) * np.sinh(k * lambda_) + k * lambda_ * np.cosh(k * lambda_))
        )

        B = numerator / Delta
        return B

    def makeC(self,poisson, k, c, A_g, eps, lambda_):
        L = self.makeL(poisson, k, c, A_g, eps, lambda_)
        M = self.makeM(poisson, k, c, A_g, eps, lambda_)
        Delta = self.makeDelta(poisson, k, lambda_)

        numerator = (
                lambda_ * L * ((1 - 2 * poisson) * np.sinh(k * lambda_) - lambda_ * k * np.cosh(k * lambda_))
                + lambda_ * M * (2 * (1 - poisson) * np.cosh(k * lambda_) + k * lambda_ * np.sinh(k * lambda_))
        )

        C = numerator / Delta
        return C

    def uHH_integrand(self, lambda_, z, rho, eps, c, poisson):
        val = lambda_ * (eps * np.exp(lambda_ * eps * (z - c)) +
                             (3 - 4 * poisson + 2 * z * lambda_) *
                             np.exp(-lambda_ * (z + c)))
        return val * j0(lambda_ * rho)

    def compute_deformation_transfer(self, z, c, k, rho, poisson, E, dV, component):
        scale = 1000 # convert to km
        # depth of receiver positions
        z = np.array(z)/scale
        rho = np.array(rho)/scale
        # depth of reservoir cell
        c = c/scale
        k = k/scale
        component = list(component)
        # number of measurement locations
        n_rec = len(z)
        assert len(rho) == n_rec
        THH = np.zeros(n_rec)
        TRB = np.zeros(n_rec)

        # Constants
        A_g = -dV * E / (4 * np.pi * (1 + poisson))
        uHH_outside_intregral = -(A_g * (1 + poisson)) / E
        uRB_outside_intregral = (1 + poisson) / E

        for c_n in component:
            if c_n == 'Geertsma_vertical':
                lambda_max = 15 / np.max(rho).item()
                for i in range(n_rec):
                    if rho[i] > np.abs(c-z[i])*3:
                        THH[i] = 0
                    else:
                        eps = np.sign(c - z[i])
                        THH[i] = quad(lambda lambda_var: self.uHH_integrand(lambda_var, z[i], rho[i], eps, c, poisson), 0, lambda_max)[0] * uHH_outside_intregral
                        THH[i] = THH[i]*scale**-2

            elif c_n == 'System_3_vertical':
                lambda_max = 30 / np.max(rho).item()
                num_points = 500
                lambda_grid = np.linspace(0, lambda_max, num_points)

                sinh_z = np.sinh(z[:, np.newaxis] * lambda_grid)
                cosh_z = np.cosh(z[:, np.newaxis] * lambda_grid)
                J0_rho = j0(lambda_grid * rho)

                #
                for i in range(n_rec):
                    if rho[i] > np.abs(c-z[i])*3:
                        TRB[i] = 0
                    else:
                        z_i = z[i]
                        sinh_z_i = sinh_z[i]
                        cosh_z_i = cosh_z[i]

                        b_values = self.makeB(poisson, k, c, A_g, -1, lambda_grid)
                        c_values = self.makeC(poisson, k, c, A_g, -1, lambda_grid)

                        part1 = b_values * (lambda_grid * z_i * cosh_z_i - (1 - 2 * poisson) * sinh_z_i)
                        part2 = c_values * ((2 * (1 - poisson) * cosh_z_i) - lambda_grid * z_i * sinh_z_i)

                        values = (part1 + part2) * J0_rho[:, i]#J0_rho_j

                        integral_result = np.trapz(values, lambda_grid)
                        TRB[i] = integral_result * uRB_outside_intregral
                        TRB[i] = TRB[i]*scale**-2

        return THH, TRB

    def h_t(self, h, r=None, k=None, i_k=None):
        """
        Hankel transform of order 0.

        Args:
        h -- Signal h(r).
        r -- Radial positions [m] (optional).
        k -- Spatial frequencies [rad/m] (optional).
        I -- Integration kernel (optional).

        Returns:
        h_t -- Spectrum H(k).
        I -- Integration kernel.
        """

        # Check if h is a vector
        if h.ndim > 1:
            raise ValueError('Signal must be a vector.')

        if r is None or len(r) == 0:
            r = np.arange(len(h))  # Default to 0:numel(h)-1
        else:
            r = np.sort(r)
            h = h[np.argsort(r)]  # Sort h according to sorted r

        if k is None or len(k) == 0:
            k = np.pi / len(h) * np.arange(len(h))  # Default spatial frequencies

        if i_k is None:
            # Create integration kernel I
            r = np.concatenate([(r[:-1] + r[1:]) / 2, [r[-1]]])  # Midpoints plus last point
            i_k = (2 * np.pi / k[:, np.newaxis]) * r * jv(1, k[:, np.newaxis] * r)  # Bessel function
            i_k[k == 0, :] = np.pi * r * r
            i_k = i_k - np.hstack([np.zeros((len(k), 1)), i_k[:, :-1]])  # Shift integration kernel
        else:
            # Ensure I is sorted based on r
            i_k = i_k[:, np.argsort(r)]

        # Compute Hankel Transform
        h_t = np.reshape(i_k @ h.flatten(), k.shape)



        return h_t, i_k

    def _get_disp_info(self, disp_config=None):
        """
        seafloor displacement (uplift/subsidence) configuration
        """
        # list of configuration parameters in the "Grav" section of teh pipt file
        config_para_list = ['baseline', 'vintage', 'method', 'model', 'poisson', 'compressibility',
                            'z_base', 'meas_spacing', 'padding', 'seabed', 'water_depth', 'baseline_mode',
                            'receiver_trunc_factor', 'geertsma_mode']

        if 'sea_disp' in self.input_dict:
            self.disp_config = {}
            for elem in self.input_dict['sea_disp']:
                assert elem[0] in config_para_list, f'Property {elem[0]} not supported'
                if elem[0] == 'vintage' and not isinstance(elem[1], list):
                    elem[1] = [elem[1]]
                self.disp_config[elem[0]] = elem[1]
            self.disp_config['baseline_mode'] = str(self.disp_config.get('baseline_mode', 'fixed')).lower()
            if self.disp_config['baseline_mode'] not in ('fixed', 'rolling'):
                raise ValueError(
                    f"Invalid SEA_DISP config: baseline_mode={self.disp_config['baseline_mode']}. "
                    "Expected 'fixed' or 'rolling'."
                )
            self.disp_config['receiver_trunc_factor'] = float(
                self.disp_config.get('receiver_trunc_factor', 3.0)
            )
            if self.disp_config['receiver_trunc_factor'] <= 0:
                raise ValueError(
                    f"Invalid SEA_DISP config: receiver_trunc_factor={self.disp_config['receiver_trunc_factor']}. "
                    "Must be > 0."
                )

            self.disp_config['geertsma_mode'] = str(
                self.disp_config.get('geertsma_mode', 'integral')
            ).lower()
            if self.disp_config['geertsma_mode'] not in ('integral', 'fast'):
                raise ValueError(
                    f"Invalid SEA_DISP config: geertsma_mode={self.disp_config['geertsma_mode']}. "
                    "Expected 'integral' or 'fast'."
                )
        else:
            self.disp_config = None

    def extract_data(self, member):
        # start by getting the data from the flow simulator i.e. prod. and inj. data
        super(flow_rock, self).extract_data(member)
        self._recover_missing_well_summary_data(member)

        # get the gravity data from results
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'sea_disp' in key:
                    if self.true_prim[1][prim_ind] in self.disp_config['vintage']:
                        v = self.disp_config['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.disp_result[v].flatten()

class flow_multi_data(flow_avo, flow_grav, flow_seafloor_disp):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        if 'grav' in input_dict.get('datatype', []):
            assert 'grav' in input_dict, 'To do GRAV simulation, please specify an "GRAV" section in the "FWDSIM" part'
            self._get_grav_info()

        if 'avo' in input_dict.get('datatype', []):
            assert 'avo' in input_dict, 'To do AVO simulation, please specify an "AVO" section in the "FWDSIM" part'
            self._get_avo_info()

        if 'sea_disp' in input_dict.get('datatype', []):
            assert 'sea_disp' in input_dict, 'To do subsidence/uplift simulation, please specify an "SEA_DISP" section in the "FWDSIM" part'
            self._get_disp_info()

    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder
        else:
            self.folder = folder

        # run flow  simulator
        # success = True
        success = super(flow_rock, self).call_sim(folder, True)

        # use output from flow simulator to forward model gravity response
        if success:
            if 'grav' in self.all_data_types:
                # calculate gravity data based on flow simulation output
                self.get_grav_result(folder, save_folder)
            if 'avo' in self.all_data_types:
                # calculate avo data based on flow simulation output
                self.get_avo_result(folder, save_folder)
            if 'sea_disp' in self.all_data_types:
                # calculate gravity data based on flow simulation output
                self.get_displacement_result(folder, save_folder)

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator i.e. prod. and inj. data
        super(flow_rock, self).extract_data(member)
        self._recover_missing_well_summary_data(member)

        # get the gravity data from results
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'grav' in key:
                    if self.true_prim[1][prim_ind] in self.grav_config['vintage']:
                        v = self.grav_config['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.grav_result[v].flatten()

                if 'avo' in key:
                    if self.true_prim[1][prim_ind] in self.pem.input['vintage']:
                        idx = self.pem.input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.avo_result[idx].flatten()

                if 'sea_disp' in key:
                    if self.true_prim[1][prim_ind] in self.disp_config['vintage']:
                        v = self.disp_config['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.disp_result[v].flatten()

