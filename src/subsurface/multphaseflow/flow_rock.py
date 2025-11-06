from subsurface.multphaseflow.opm import flow
from importlib import import_module
import datetime as dt
import numpy as np
import os
from subsurface.multphaseflow.misc import ecl, grdecl
import shutil
import glob
import io
import re
import warnings
from subprocess import Popen, PIPE
import mat73
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile

from resdata.summary import Summary
from resdata.resfile import ResdataRestartFile,ResdataInitFile
from resdata.grid import Grid

# Import heavy dependencies for flow_equinor_sim2seis at module level
try:
    import pandas as pd
    import open_petro_elastic as pem
    from open_petro_elastic.__main__ import make_input
    _PEM_AVAILABLE = True
except ImportError:
    pd = None
    pem = None
    make_input = None
    _PEM_AVAILABLE = False


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

            pem = getattr(import_module('simulator.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)

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
            # need a if to check that we have correct sim2seis
            # copy relevant sim2seis files into folder.
            for file in glob.glob('sim2seis_config/*'):
                shutil.copy(file, 'En_' + str(self.ensemble_member) + os.sep)

            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')
            grid = self.ecl_case.grid()

            case_name = 'En_' + str(self.ensemble_member) + os.sep + self.file
            restart = ResdataRestartFile(Grid(case_name), f"{case_name}.UNRST")
            vintages = restart.report_dates

            phases = self.ecl_case.init.phases
            if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
                # loop over seismic vintages
                for v, time in enumerate(vintages):
                    pem_input = {}
                    # get active porosity
                    tmp = self.ecl_case.cell_data('PORO')
                    if 'compaction' in self.pem_input:
                        multfactor = self.ecl_case.cell_data('PORV_RC', time)

                        pem_input['PORO'] = np.array(
                            multfactor[~tmp.mask]*tmp[~tmp.mask], dtype=float)
                    else:
                        pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)
                    # get active NTG if needed
                    if 'ntg' in self.pem_input:
                        if self.pem_input['ntg'] == 'no':
                            pem_input['NTG'] = None
                        else:
                            tmp = self.ecl_case.cell_data('NTG')
                            pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
                    else:
                        tmp = self.ecl_case.cell_data('NTG')
                        pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)

                    pem_input['PRESSURE'] = restart.restart_get_kw('PRESSURE',time).numpy_copy()*0.1 # Bar to MPa
                    for var in ['PRESSURE','SWAT', 'SGAS', 'RS']:
                        try:
                            tmp = restart.restart_get_kw(var,time).numpy_copy()
                            # only active, and conv. to float
                            pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)
                        except:
                            pem_input[var] = np.zeros_like(pem_input['PRESSURE'], dtype=float) # This is always here.

                    if 'press_conv' in self.pem_input:
                        pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                            self.pem_input['press_conv']

                    tmp = restart.restart_get_kw('PRESSURE',vintages[0]).numpy_copy()*0.1 # Bar to MPa
                    if hasattr(self.pem, 'p_init'):
                        P_init = self.pem.p_init*np.ones(tmp.shape)[~tmp.mask]
                    else:
                        # initial pressure is first
                        P_init = np.array(tmp[~tmp.mask], dtype=float)

                    if 'press_conv' in self.pem_input:
                        P_init = P_init*self.pem_input['press_conv']

                    saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                                   for ph in phases]
                    # Get the pressure
                    self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                        ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                        ensembleMember=self.ensemble_member)

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
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        result = mat73.loadmat(f'En_{member}/Data_conv.mat')['data_conv']
                        v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = np.sum(
                            np.abs(result[:, :, :, v]), axis=0).flatten()

class flow_rock(flow):
    """
    Couple the OPM-flow simulator with a rock-physics simulator such that both reservoir quantities and petro-elastic
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

            pem = getattr(import_module('subsurface.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)

        else:
            self.pem = None

    def setup_fwd_run(self, **kwargs):
        self.__dict__.update(kwargs)  # parse kwargs input into class attributes
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
            case_name = 'En_' + str(self.ensemble_member) + os.sep + self.file
            restart = ResdataRestartFile(Grid(case_name), f"{case_name}.UNRST")
            data_vintages = restart.report_dates

            self.pem_input['vintage'] = data_vintages[1:] #hardcode this

            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')

            #phases = self.ecl_case.init.phases
            phases = ['OIL','WAT','GAS'] #self.ecl_case.init.phases
            #if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            #if 'WAT' in phases and 'GAS' in phases:
            vintage = []
            # loop over seismic vintages
            for v, time in enumerate(data_vintages[1:]):
                pem_input = {}
                # get active porosity
                tmp = self.ecl_case.cell_data('PORO')
                if 'compaction' in self.pem_input:
                    multfactor = self.ecl_case.cell_data('PORV_RC', time)

                    pem_input['PORO'] = np.array(
                        multfactor[~tmp.mask]*tmp[~tmp.mask], dtype=float)
                else:
                    pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)
                # get active NTG if needed
                if 'ntg' in self.pem_input:
                    if self.pem_input['ntg'] == 'no':
                        pem_input['NTG'] = None
                    else:
                        tmp = self.ecl_case.cell_data('NTG')
                        pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
                else:
                    tmp = self.ecl_case.cell_data('NTG')
                    pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)

                pem_input['RS'] = None
                pem_input['PRESSURE'] = restart.restart_get_kw('PRESSURE', time).numpy_copy() * 0.1  # Bar to MPa

                for var in ['SWAT', 'SGAS', 'RS']:
                    try:
                        tmp = restart.restart_get_kw(var,time).numpy_copy()
                    except:
                        tmp = np.zeros_like(pem_input['PRESSURE'], dtype=float)
                    # only active, and conv. to float
                    pem_input[var] = tmp

                if 'press_conv' in self.pem_input:
                    pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                        self.pem_input['press_conv']

                tmp = restart.restart_get_kw('PRESSURE', data_vintages[0]).numpy_copy()*0.1 # Bar to MPa
                if hasattr(self.pem, 'p_init'):
                    P_init = self.pem.p_init*np.ones(tmp.shape)[~tmp.mask]
                else:
                    # initial pressure is first
                    P_init = tmp

                if 'press_conv' in self.pem_input:
                    P_init = P_init*self.pem_input['press_conv']

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
                base_time = data_vintages[int(self.pem.baseline)]
                #base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                #                        self.startDate['day']) + dt.timedelta(days=self.pem.baseline)
                # pem_input = {}
                # get active porosity
                tmp = self.ecl_case.cell_data('PORO')

                if 'compaction' in self.pem_input:
                    multfactor = self.ecl_case.cell_data('PORV_RC', base_time)

                    pem_input['PORO'] = np.array(
                        multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
                else:
                    pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)

                pem_input['RS'] = None
                pem_input['PRESSURE'] = restart.restart_get_kw('PRESSURE',base_time).numpy_copy()*0.1 # Bar to MPa
                for var in ['SWAT', 'SGAS', 'RS']:
                    try:
                        tmp = self.ecl_case.cell_data(var, base_time)
                    except:
                        tmp = np.zeros_like(pem_input['PRESSURE'])
                    # only active, and conv. to float
                    pem_input[var] = tmp

                if 'press_conv' in self.pem_input:
                    pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                        self.pem_input['press_conv']

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

        # get the sim2seis from file
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if key in ['bulkimp']:
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.pem_result[v].data.flatten()

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
                for v, assim_time in enumerate(self.pem_input['vintage']):
                    time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                        dt.timedelta(days=assim_time)
                    pem_input = {}
                    # get active porosity
                    tmp = self.ecl_case.cell_data('PORO')
                    if 'compaction' in self.pem_input:
                        multfactor = self.ecl_case.cell_data('PORV_RC', time)

                        pem_input['PORO'] = np.array(
                            multfactor[~tmp.mask]*tmp[~tmp.mask], dtype=float)
                    else:
                        pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)
                    # get active NTG if needed
                    if 'ntg' in self.pem_input:
                        if self.pem_input['ntg'] == 'no':
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

                    if 'press_conv' in self.pem_input:
                        pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                            self.pem_input['press_conv']

                    tmp = self.ecl_case.cell_data('PRESSURE', 1)
                    if hasattr(self.pem, 'p_init'):
                        P_init = self.pem.p_init*np.ones(tmp.shape)[~tmp.mask]
                    else:
                        # initial pressure is first
                        P_init = np.array(tmp[~tmp.mask], dtype=float)

                    if 'press_conv' in self.pem_input:
                        P_init = P_init*self.pem_input['press_conv']

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

                if 'compaction' in self.pem_input:
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

                if 'press_conv' in self.pem_input:
                    pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                        self.pem_input['press_conv']

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
            if 'clusters' in self.pem_input:
                npzfile = np.load(self.pem_input['clusters'], allow_pickle=True)
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
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.bar_result[v].flatten()


class flow_equinor_sim2seis(flow):
    """
    Run OPM / Eclipse simulations and derive sim2seis style petro-elastic responses
    using only Eclipse binary files. This mirrors the :class:`flow_rock` workflow,
    but utilises the open_petro_elastic package to generate Vp/Vs/AI/SI directly.
    
    The class calculates 4D differences between target times and a baseline time.
    If a baseline time is specified, the results will be the difference between
    the target time properties and the baseline time properties. If no baseline
    is specified, absolute values are returned.
    """

    PROPS_DYNAMIC = ['PRESSURE', 'RS', 'SWAT', 'SGAS']
    PROPS_STATIC = ['FSAND', 'PORO', 'dZ', 'PRESSURE_INIT']

    def __init__(self, input_dict=None, filename=None, options=None):
        input_dict = input_dict or {}
        super().__init__(input_dict, filename, options)
        self._root_path = os.getcwd()

        self._configure_pem(input_dict)

        self.dum_file_root = 'dummy.txt'
        self.dum_entry = str(0)
        self.date_slack = None
        if 'date_slack' in input_dict:
            self.date_slack = int(input_dict['date_slack'])

        self.saveinfo = None
        if 'savesiminfo' in input_dict:
            if isinstance(input_dict['savesiminfo'], list):
                self.saveinfo = input_dict['savesiminfo']
            else:
                self.saveinfo = [input_dict['savesiminfo']]

        self.scale = []
        self._pem_results = {}
        self._pem_dataframe = None
        self._target_times = []
        self._baseline_time = None
        self._warned_flag_model = False

    def _configure_pem(self, input_dict):
        self.pem_input = {}
        if 'pem' not in input_dict:
            return

        for elem in input_dict['pem']:
            if elem[0] == 'config':
                self.pem_input['config'] = elem[1]
            if elem[0] == 'model':
                self.pem_input['model'] = elem[1]
            if elem[0] == 'baseline':
                self.pem_input['baseline'] = elem[1]
            if elem[0] == 'vintage':
                self.pem_input['vintage'] = elem[1] if isinstance(elem[1], list) else [elem[1]]
            if elem[0] == 'ntg':
                self.pem_input['ntg'] = elem[1]
            if elem[0] == 'press_conv':
                self.pem_input['press_conv'] = elem[1]

    def setup_fwd_run(self, **kwargs):
        self.__dict__.update(kwargs)
        super().setup_fwd_run()

    def run_fwd_sim(self, state, member_i, del_folder=True):
        self.ensemble_member = member_i
        self._pem_results = {}
        self._pem_dataframe = None
        self._target_times = []
        self._baseline_time = None
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=del_folder)
        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False):
        success = super().call_sim(folder, wait_for_proc)
        if not success:
            return success

        if not _PEM_AVAILABLE:
            raise RuntimeError(
                "flow_eclipse_sim2seis requires pandas and open_petro_elastic to be installed"
            )

        member_folder = os.path.join('En_' + str(self.ensemble_member))
        case_name = member_folder + os.sep + self.file

        self.ecl_case = ecl.EclipseCase(case_name + '.DATA')
        restart = ResdataRestartFile(Grid(case_name), f"{case_name}.UNRST")
        report_dates = restart.report_dates

        self._target_times = self._resolve_target_times(report_dates)
        self._baseline_time = self._resolve_baseline_time(report_dates)

        # Include baseline time in data collection if it's not already in target times
        times_to_collect = self._target_times.copy()
        if self._baseline_time is not None and self._baseline_time not in times_to_collect:
            times_to_collect.append(self._baseline_time)
        
        baseline_grid = Grid(case_name + '.EGRID')
        df = self._collect_eclipse_frame_optimized(restart, report_dates, times_to_collect)
        self._pem_dataframe = df
        self._pem_results = self._run_pem_dataframe_optimized(
            df, self._target_times, member_folder, baseline_grid
        )

        if self.saveinfo is not None:
            try:
                from pipt.misc_tools.analysis_tools import store_ensemble_sim_information
                store_ensemble_sim_information(self.saveinfo, self.ensemble_member)
            except Exception:
                pass

        return success

    def extract_data(self, member):
        super().extract_data(member)
        if not self._pem_results:
            return

        for prim_ind in self.l_prim:
            date_token = self._format_date(self.true_prim[1][prim_ind])
            outputs = self._pem_results.get(date_token)
            if outputs is None:
                continue

            for name, values in outputs.items():
                candidates = [
                    name,
                    name.upper(),
                    name.lower(),
                    'PEM_' + name.upper(),
                ]
                assigned = False
                for candidate in candidates:
                    if candidate in self.pred_data[prim_ind]:
                        self.pred_data[prim_ind][candidate] = values
                        assigned = True
                        break
                if not assigned and name == 'AI' and 'sim2seis' in self.pred_data[prim_ind]:
                    self.pred_data[prim_ind]['sim2seis'] = values

    def _resolve_target_times(self, report_dates):
        if not isinstance(report_dates, list):
            report_dates = list(report_dates)

        mapping = {self._format_date(date): date for date in report_dates}
        vintages = self.pem_input.get('vintage', [])
        resolved = []
        for entry in vintages:
            token = self._format_date(entry)
            if token in mapping:
                resolved.append(mapping[token])

        if resolved:
            return resolved

        if len(report_dates) > 1:
            return report_dates[1:]
        return report_dates

    def _resolve_baseline_time(self, report_dates):
        if not report_dates:
            return None

        mapping = {self._format_date(date): date for date in report_dates}
        baseline = self.pem_input.get('baseline')
        if baseline is not None:
            token = self._format_date(baseline)
            if token in mapping:
                return mapping[token]
            if isinstance(baseline, int) and 0 <= baseline < len(report_dates):
                return report_dates[baseline]
        return report_dates[0]

    def _collect_eclipse_frame_optimized(self, restart, report_dates, target_times):
        """Optimized version focusing only on required properties for speed.
        Collects data for target times and baseline time if needed."""
        
        # Extract static properties once
        poro = self._masked_values(self.ecl_case.cell_data('PORO'))
        
        try:
            ntg = self._masked_values(self.ecl_case.cell_data('NTG'))
        except Exception:
            ntg = np.ones_like(poro)
        
        # Only get depth if absolutely necessary (skip if not used in PEM)
        try:
            dz = self._masked_values(self.ecl_case.cell_data('DZ'))
        except Exception:
            dz = np.ones_like(poro)
        
        # Pre-allocate data dictionary with known size
        n_cells = len(poro)
        
        # Determine all times we need to collect (target times plus baseline)
        all_times = set(target_times)
        if self._baseline_time is not None:
            all_times.add(self._baseline_time)
        all_times = list(all_times)
        
        n_times = len(all_times)
        
        # Static data
        data = {
            'PORO': np.clip(poro, 0.0, 0.3),  # Apply clipping once
            'NTG': ntg,
            'dZ': dz
        }
        
        # Calculate FSAND once from static properties
        vsh = 1.0 - ntg
        denom = np.maximum(1.0 - poro, 1.0e-8)
        fsand = 1.0 - vsh / denom
        data['FSAND'] = np.clip(fsand, 0.0, 1.0)
        
        # Get baseline pressure once
        if self._baseline_time is not None:
            base_pressure = self._restart_kw_values(restart, 'PRESSURE', self._baseline_time)
        else:
            base_pressure = self._restart_kw_values(restart, 'PRESSURE', report_dates[0])
        
        base_pressure = base_pressure * 1.0e5
        if 'press_conv' in self.pem_input:
            base_pressure = base_pressure * self.pem_input['press_conv']
        data['PRESSURE_INIT'] = base_pressure
        
        # Pre-allocate arrays for dynamic properties to avoid repeated allocations
        pressure_conv = self.pem_input.get('press_conv', 1.0) * 1.0e5
        epsilon = 1.0e-4
        
        # Extract dynamic properties for each time step (including baseline)
        for time in all_times:
            token = self._format_date(time)
            
            # Extract pressure and apply conversion in one step
            pressure = self._restart_kw_values(restart, 'PRESSURE', time) * pressure_conv
            data[f'PRESSURE_{token}'] = pressure
            
            # Extract saturation properties with error handling
            for keyword in ['RS', 'SWAT', 'SGAS']:
                try:
                    values = self._restart_kw_values(restart, keyword, time)
                except Exception:
                    values = np.zeros(n_cells, dtype=np.float32)  # Use float32 for memory efficiency
                data[f'{keyword}_{token}'] = values
        
        # Create DataFrame once with all data
        df = pd.DataFrame(data)
        
        # Apply saturation constraints efficiently
        for time in all_times:
            token = self._format_date(time)
            swat_col = f'SWAT_{token}'
            sgas_col = f'SGAS_{token}'
            
            if swat_col in df.columns and sgas_col in df.columns:
                # Vectorized saturation constraint application
                swat = np.clip(df[swat_col].values, 0.0, 1.0 - epsilon / 2)
                sgas = np.maximum(0.0, np.minimum(df[sgas_col].values, 1.0 - swat - epsilon / 2))
                df[swat_col] = swat
                df[sgas_col] = sgas
        
        return df

    def _collect_eclipse_frame(self, restart, report_dates, target_times):
        """Collect eclipse data for target times and baseline time if needed."""

        data = {}
        data['PORO'] = self._masked_values(self.ecl_case.cell_data('PORO'))

        try:
            data['NTG'] = self._masked_values(self.ecl_case.cell_data('NTG'))
        except Exception:
            data['NTG'] = np.ones_like(data['PORO'])

        depth = None
        for keyword in ['DEPTH', 'DEPT', 'DEPTZ', 'DEPTHZ']:
            try:
                depth = self._masked_values(self.ecl_case.cell_data(keyword))
                break
            except Exception:
                continue
        if depth is None:
            depth = np.zeros_like(data['PORO'])
        data['DEPTH'] = depth

        try:
            data['dZ'] = self._masked_values(self.ecl_case.cell_data('DZ'))
        except Exception:
            data['dZ'] = np.ones_like(data['PORO'])

        if self._baseline_time is not None:
            base_pressure = self._restart_kw_values(restart, 'PRESSURE', self._baseline_time)
        else:
            base_pressure = self._restart_kw_values(restart, 'PRESSURE', report_dates[0])
        base_pressure = base_pressure * 1.0e5
        if 'press_conv' in self.pem_input:
            base_pressure = base_pressure * self.pem_input['press_conv']
        data['PRESSURE_INIT'] = base_pressure

        # Collect data for all required times (target times and baseline if not included)
        all_times = set(target_times)
        if self._baseline_time is not None:
            all_times.add(self._baseline_time)

        for time in all_times:
            token = self._format_date(time)
            pressure = self._restart_kw_values(restart, 'PRESSURE', time) * 1.0e5
            if 'press_conv' in self.pem_input:
                pressure = pressure * self.pem_input['press_conv']
            data['PRESSURE_' + token] = pressure

            for keyword in ['RS', 'SWAT', 'SGAS']:
                try:
                    values = self._restart_kw_values(restart, keyword, time)
                except Exception:
                    values = np.zeros_like(data['PORO'])
                data[f'{keyword}_{token}'] = values

        df = pd.DataFrame(data)

        epsilon = 1.0e-4
        for time in all_times:
            token = self._format_date(time)
            swat_col = 'SWAT_' + token
            sgas_col = 'SGAS_' + token
            if swat_col in df and sgas_col in df:
                swat = np.clip(df[swat_col], 0.0, 1.0 - epsilon / 2)
                sgas = np.maximum(0.0, np.minimum(df[sgas_col], 1.0 - swat - epsilon / 2))
                df[swat_col] = swat
                df[sgas_col] = sgas

        vsh = 1.0 - df['NTG']
        denom = np.maximum(1.0 - df['PORO'], 1.0e-8)
        fsand = 1.0 - vsh / denom
        df['FSAND'] = np.clip(fsand, 0.0, 1.0)
        df['PORO'] = np.clip(df['PORO'], 0.0, 0.3)

        df = df[df['DEPTH'].notna()].reset_index(drop=True)

        return df

    def _run_pem_dataframe_optimized(self, dataframe, target_times, member_folder,grid):
        """Optimized version that minimizes file I/O and memory allocations.
        Returns 4D differences between target times and baseline time."""
        config_path = self._resolve_config_path()
        results = {}

        # Get mapping from active to global indices
        active_df = grid.export_index(active_only=True)
        active_global_indices = active_df.index.to_numpy()
        
        # Pre-extract static frame once
        static_frame = dataframe[self.PROPS_STATIC].copy()
        
        # First, calculate baseline properties if baseline time is available
        baseline_props = None
        if self._baseline_time is not None:
            baseline_token = self._format_date(self._baseline_time)
            baseline_props = self._calculate_pem_properties(
                dataframe, static_frame, self._baseline_time, baseline_token, 
                config_path, member_folder
            )
        
        # Calculate properties for each target time and compute 4D differences
        for time in target_times:
            # Create full-size array with default values for inactive cells
            token = self._format_date(time)
            
            # Calculate target properties
            target_props = self._calculate_pem_properties(
                dataframe, static_frame, time, token, config_path, member_folder
            )
            
            # If we have baseline properties, calculate 4D differences
            if baseline_props is not None:
                diff_props = {}
                for prop_name, target_values in target_props.items():
                    full_array = np.full(grid.get_global_size(), 0.0)  # use 0.0 as default
                    if prop_name in baseline_props:
                        full_array[active_global_indices] = baseline_props[prop_name] - target_values
                        # scale the full array between -1 and 1
                        # find max and min for scaling
                        min_val, max_val = full_array.min(), full_array.max()
                        if max_val > min_val:
                            full_array = np.interp(full_array, (min_val, max_val), (-1, 1))
                        else:
                            full_array = np.zeros_like(full_array)

                        diff_props[prop_name] = full_array
                    else:
                        full_array[active_global_indices] = target_values
                        # Normalize the full array between 0 and 1
                        min_val = np.min(full_array)
                        max_val = np.max(full_array)
                        if max_val > min_val:
                            full_array = np.interp(full_array, (min_val, max_val), (-1, 1))
                        else:
                            full_array = np.zeros_like(full_array)
                        diff_props[prop_name] = full_array
                results[token] = diff_props
            else:
                # No baseline available, return absolute values
                results[token] = target_props
        
        return results
    
    def _calculate_pem_properties(self, dataframe, static_frame, time, token, config_path, member_folder):
        """Helper method to calculate PEM properties for a given time."""
        # Build frame more efficiently using direct column assignment
        frame = static_frame.copy()
        for prop in self.PROPS_DYNAMIC:
            column = f'{prop}_{token}'
            if column not in dataframe.columns:
                raise KeyError(f'Missing column {column} for PEM input')
            frame[prop] = dataframe[column].values  # Use .values for faster copying
        
        # Use traditional temporary file approach but optimized
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, dir=member_folder) as tmp_csv:
            frame.to_csv(tmp_csv.name, index=False)
            csv_path = tmp_csv.name
        
        try:
            pem_input = self._load_pem_input(config_path, csv_path)
        finally:
            try:
                os.unlink(csv_path)
            except OSError:
                pass
        
        # PEM calculations (this is the computational bottleneck, but required)
        pressure = pem_input.pressure
        minerals = pem_input.minerals.as_mixture
        fluids = pem_input.fluids.as_mixture(pressure)
        dry_rock = pem_input.dry_rock.material(minerals, pressure)
        dense = pem_input.dry_rock.nonporous(minerals)
        saturated = pem.material.fluid_substitution(
            dry_rock,
            dense,
            fluids,
            pem_input.dry_rock.porosity,
        )
        
        # Extract results efficiently using direct numpy array conversion
        vp = saturated.primary_velocity.astype(np.float32, copy=False)
        vs = saturated.secondary_velocity.astype(np.float32, copy=False)
        ai = saturated.acoustic_impedance.astype(np.float32, copy=False)
        # Replace NaN values with mean of valid numbers
        ai = np.where(np.isnan(ai), np.nanmean(ai), ai)
        si = saturated.shear_impedance.astype(np.float32, copy=False)
        dens = saturated.density.astype(np.float32, copy=False)
        
        # Vectorized calculations
        vpvs = vp / vs
        dz = frame['dZ'].values.astype(np.float32)
        dtpp = 2000.0 * dz / vp
        dtss = 2000.0 * dz / vs
        dtps = (dtpp + dtss) * 0.5  # Faster than division by 2
        
        return {
            'VP': vp,
            'VS': vs,
            'AI': ai,
            'SI': si,
            'DENS': dens,
            'VPVS': vpvs,
            'DTPP': dtpp,
            'DTSS': dtss,
            'DTPS': dtps,
        }

    def _run_pem_dataframe(self, dataframe, target_times, member_folder, pem, make_input, tempfile):
        """Calculate PEM properties and return 4D differences between target times and baseline time."""
        config_path = self._resolve_config_path()
        results = {}
        static_frame = dataframe[self.PROPS_STATIC].copy()

        # First, calculate baseline properties if baseline time is available
        baseline_props = None
        if self._baseline_time is not None:
            baseline_token = self._format_date(self._baseline_time)
            baseline_props = self._calculate_pem_properties_legacy(
                dataframe, static_frame, self._baseline_time, baseline_token,
                config_path, member_folder, tempfile
            )

        # Calculate properties for each target time and compute 4D differences
        for time in target_times:
            token = self._format_date(time)
            
            # Calculate target properties
            target_props = self._calculate_pem_properties_legacy(
                dataframe, static_frame, time, token, config_path, member_folder, tempfile
            )
            
            # If we have baseline properties, calculate 4D differences
            if baseline_props is not None:
                diff_props = {}
                for prop_name, target_values in target_props.items():
                    if prop_name in baseline_props:
                        diff_props[prop_name] = target_values - baseline_props[prop_name]
                    else:
                        diff_props[prop_name] = target_values
                results[token] = diff_props
            else:
                # No baseline available, return absolute values
                results[token] = target_props

        return results
    
    def _calculate_pem_properties_legacy(self, dataframe, static_frame, time, token, config_path, member_folder, tempfile):
        """Helper method to calculate PEM properties for a given time (legacy version)."""
        frame = static_frame.copy()
        for prop in self.PROPS_DYNAMIC:
            column = f'{prop}_{token}'
            if column not in dataframe.columns:
                raise KeyError(f'Missing column {column} for PEM input')
            frame[prop] = dataframe[column]

        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, dir=member_folder) as tmp_csv:
            frame.to_csv(tmp_csv.name, index=False)
            csv_path = tmp_csv.name

        pem_input = self._load_pem_input(config_path, csv_path)

        try:
            os.unlink(csv_path)
        except OSError:
            pass

        pressure = pem_input.pressure
        minerals = pem_input.minerals.as_mixture
        fluids = pem_input.fluids.as_mixture(pressure)
        dry_rock = pem_input.dry_rock.material(minerals, pressure)
        dense = pem_input.dry_rock.nonporous(minerals)
        saturated = pem.material.fluid_substitution(
            dry_rock,
            dense,
            fluids,
            pem_input.dry_rock.porosity,
        )

        vp = np.asarray(saturated.primary_velocity, dtype=float)
        vs = np.asarray(saturated.secondary_velocity, dtype=float)
        ai = np.asarray(saturated.acoustic_impedance, dtype=float)
        si = np.asarray(saturated.shear_impedance, dtype=float)
        dens = np.asarray(saturated.density, dtype=float)
        vpvs = vp / vs

        dz = frame['dZ'].to_numpy(dtype=float)
        dtpp = 2000.0 * dz / vp
        dtss = 2000.0 * dz / vs
        dtps = (dtpp + dtss) / 2.0

        return {
            'VP': vp,
            'VS': vs,
            'AI': ai,
            'SI': si,
            'DENS': dens,
            'VPVS': vpvs,
            'DTPP': dtpp,
            'DTSS': dtss,
            'DTPS': dtps,
        }

    def _load_pem_input(self, config_path, csv_path):
        stream = self._normalized_pem_config_stream(config_path)
        try:
            return make_input(stream, csv_path)
        finally:
            stream.close()

    def _normalized_pem_config_stream(self, config_path):
        with open(config_path, 'r') as cfg_stream:
            config_text = cfg_stream.read()
        normalized_text = self._normalize_pem_config_text(config_text)
        return io.StringIO(normalized_text)

    def _normalize_pem_config_text(self, config_text):
        normalized = config_text
        replacement_done = False

        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None

        if yaml is not None:
            try:
                config_data = yaml.safe_load(config_text)
            except yaml.YAMLError:
                config_data = None
            if isinstance(config_data, dict):
                fluids = config_data.get('fluids')
                if isinstance(fluids, dict):
                    model = fluids.get('fluid_model')
                    if model == 'flag':
                        fluids['fluid_model'] = 'batzle_wang'
                        normalized = yaml.safe_dump(
                            config_data,
                            sort_keys=False,
                            default_flow_style=False,
                        )
                        replacement_done = True
                        self._warn_flag_model()

        if not replacement_done:
            pattern = re.compile(r'(fluid_model\s*:\s*)(flag)\b')
            if pattern.search(config_text):
                normalized = pattern.sub(r'\1batzle_wang', config_text)
                self._warn_flag_model()

        return normalized

    def _warn_flag_model(self):
        if not self._warned_flag_model:
            warnings.warn(
                "PEM configuration requested fluid_model 'flag', which is not available. "
                "Using 'batzle_wang' instead.",
                RuntimeWarning,
            )
            self._warned_flag_model = True

    def _resolve_config_path(self):
        config_path = self.pem_input.get('config')
        if config_path is None:
            raise RuntimeError("Missing PEM configuration. Provide 'config' in pem input.")
        config_path = os.path.expanduser(config_path)
        if not os.path.isabs(config_path):
            config_path = os.path.join(self._root_path, config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'PEM configuration file not found: {config_path}')
        return config_path

    def _masked_values(self, array):
        if hasattr(array, 'compressed'):
            return np.asarray(array.compressed(), dtype=float)
        if hasattr(array, 'mask'):
            return np.asarray(array[~array.mask], dtype=float)
        return np.asarray(array, dtype=float).reshape(-1)

    def _restart_kw_values(self, restart, keyword, time):
        values = restart.restart_get_kw(keyword, time).numpy_copy()
        if hasattr(values, 'compressed'):
            return np.asarray(values.compressed(), dtype=float)
        if hasattr(values, 'mask'):
            return np.asarray(values[~values.mask], dtype=float)
        return np.asarray(values, dtype=float).reshape(-1)

    def _format_date(self, value):
        if isinstance(value, dt.datetime):
            return value.strftime('%Y%m%d')
        if isinstance(value, str):
            return value.replace('-', '')
        if isinstance(value, (np.integer, int)):
            return f'{int(value):08d}'
        return str(value)
