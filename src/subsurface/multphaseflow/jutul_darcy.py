'''
Simulator wrapper for the JutulDarcy simulator.

This module provides a wrapper interface for running JutulDarcy simulations
with support for ensemble-based workflows and flexible output formatting.
'''

#────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
import shutil
import os

from mako.template import Template
from typing import Union
from p_tqdm import p_map
from tqdm import tqdm
#────────────────────────────────────────────────────


__author__ = 'Mathias Methlie Nilsen'
__all__ = ['JutulDarcyWrapper']


#────────────────────────────────────────────────────────────────────────────────────
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
os.environ['PYTHON_JULIACALL_THREADS'] = '1'
os.environ['PYTHON_JULIACALL_OPTLEVEL'] = '3'
warnings.filterwarnings('ignore', message='.*juliacall module already imported.*')
#────────────────────────────────────────────────────────────────────── ──────────────

PBAR_OPTS = {
    'ncols': 110,
    'colour': "#285475",
    'bar_format': '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    'ascii': '-◼', # Custom bar characters for a sleeker look
}


class JutulDarcyWrapper:

    def __init__(self, options):
        '''
        Wrapper for the JutulDarcy simulator [1].

        Parameters
        ----------
        options : dict
            Configuration options for the wrapper.
            Keys:
                - 'makofile' or 'runfile': Path to the makofile.mako or runfile.DATA template.
                - 'reporttype': Type of report (default: 'days').
                - 'out_format': Output format ('list', 'dict', 'dataframe'; default: 'list').
                - 'datatype': List of data types to extract (default: ['FOPT', 'FGPT', 'FWPT', 'FWIT']).
                - 'parallel': Number of parallel simulations (default: 1).

        References
        ----------
        [1] Møyner, O. (2025).
            JutulDarcy.jl - a fully differentiable high-performance reservoir simulator
            based on automatic differentiation. Computational Geosciences, 29, Article 30.
            https://doi.org/10.1007/s10596-025-10366-6
        '''
        # Make makofile an mandatory option
        if ('makofile' not in options) and ('runfile' not in options):
            raise ValueError('Wrapper  requires a makofile (or runfile) option')
        
        if 'makofile' in options: 
            self.makofile = options.get('makofile')

        if 'runfile' in options:
            self.makofile = options.get('runfile').split('.')[0] + '.mako'

        # Other variables
        self.reporttype = options.get('reporttype', 'days')
        self.out_format = options.get('out_format', 'list')
        self.datatype   = options.get('datatype', ['FOPT', 'FGPT', 'FWPT', 'FWIT'])
        self.parallel   = options.get('parallel', 1)
        self.units      = options.get('units', 'metric') # This is not consistently used!

        self.datafile = None
        self.compute_adjoints = False

        # This is for PET to work properly (should be removed in future versions)
        self.input_dict = options
        self.true_order = [self.reporttype, options['reportpoint']]
        self.steps = [i+1 for i in range(len(self.true_order[1]))]

        # Adjoint settings
        #---------------------------------------------------------------------------------------------------------
        if 'adjoints' in options:
            self.compute_adjoints = True

            self.adjoint_info = {}
            for datatype in options['adjoints']:
                
                # Determine if rate or volume and phase. 
                if datatype in ['WOPT', 'WGPT', 'WWPT', 'WLPT']:
                    rate  = False
                    phase = {
                        'WOPT': 'oil',
                        'WGPT': 'gas',
                        'WWPT': 'water',
                        'WLPT': 'liquid',
                    }[datatype]
                
                elif datatype in ['WOPR', 'WGPR', 'WWPR', 'WLPR']:
                    rate  = True
                    phase = {
                        'WOPR': 'oil',
                        'WGPR': 'gas',
                        'WWPR': 'water',
                        'WLPR': 'liquid',
                    }[datatype]


                # Determine steps
                steps = options['adjoints'][datatype].get('steps', 'acc')
                if steps == 'acc':
                    steps = [self.steps[-1]]
                    accumulative = True
                elif steps == 'all':
                    steps = self.steps
                    accumulative = False
                elif isinstance(steps, int):
                    accumulative = False
                    steps = [steps]
                
                well_ids = options['adjoints'][datatype]['well_id']
                parameters = options['adjoints'][datatype]['parameters']

                # Ensure well_ids and parameters are lists
                well_ids = well_ids if isinstance(well_ids, (list, tuple)) else [well_ids]
                parameters = parameters if isinstance(parameters, (list, tuple)) else [parameters]

                for wid in well_ids:
                    self.adjoint_info[f'{datatype}:{wid}'] = {
                        'rate': rate,
                        'phase': phase,
                        'well_id': wid,
                        'parameters': parameters,
                        'steps': steps,
                        'accumulative': accumulative,
                    }
        #---------------------------------------------------------------------------------------------------------

    def __call__(self, inputs: dict):

        # Delet all existing En_* folders
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.startswith('En_'):
                shutil.rmtree(item)
        
        # simulate all inputs in parallel
        outputs = p_map(
            self.run_fwd_sim, 
            [inputs[n] for n in range(len(inputs))], 
            list(range(len(inputs))), 
            num_cpus=self.parallel,
            unit='sim',
            **PBAR_OPTS
        )

        if self.compute_adjoints:
            results, adjoints = zip(*outputs)
            return results, adjoints
        else:
            return outputs
                     

    def run_fwd_sim(self, input: dict, idn: int=0, delete_folder: bool=True):
        '''
        Run forward simulation for given input parameters.

        Parameters
        ----------
        input: dict
            Input parameters for the simulation.

        idn: int, optional
            Ensemble member ID, by default 0.

        delete_folder: bool, optional
            Whether to delete the simulation folder after running, by default True.

        Returns
        -------
            output: Union[dict, list, pd.DataFrame]
                Simulation output in the specified format.
        '''
        from juliacall import Main as jl
        from jutuldarcy import convert_to_pydict
        jl.seval("using JutulDarcy, Jutul")

        # Include ensemble member id in input dict
        input['member'] = idn
        
        # Make simulation folder
        folder = f'En_{idn}'
        os.makedirs(folder)

        # Render makofile
        self.render_makofile(self.makofile, folder, input)

        # Enter simulation folder and run simulation
        os.chdir(folder)

        # Setup case
        case = jl.setup_case_from_data_file(self.datafile)

        # Get some grid info
        nx = case.input_data["GRID"]["cartDims"][0]
        ny = case.input_data["GRID"]["cartDims"][1]  
        nz = case.input_data["GRID"]["cartDims"][2]
        grid = (nx, ny, nz)
        actnum = np.array(case.input_data["GRID"]["ACTNUM"]) # Shape (nx, ny, nz)
        actnum_vec = actnum.flatten(order='F')  # Fortran order flattening

        # Simulate and get results
        jlres = jl.simulate_reservoir(case, info_level=-1)
        pyres = convert_to_pydict(jlres, case, units=self.units)

        if self.compute_adjoints:

            # Initialize adjoint dataframe
            colnames = []
            for key in self.adjoint_info:
                for param in self.adjoint_info[key]['parameters']:
                    colnames.append((key, param))

            adjoints = pd.DataFrame(columns=pd.MultiIndex.from_tuples(colnames), index=self.true_order[1])
            adjoints.index.name = self.true_order[0]

            # Initialize progress bar
            pbar = tqdm(
                adjoints.keys(), 
                desc=f'Solving adjoints for En_{idn}',
                position=idn+1,
                leave=False,
                unit='obj',
                dynamic_ncols=False,
                **PBAR_OPTS
            )
            
            # Loop over adjoint objectives
            for col in adjoints.columns.levels[0]:
                info = self.adjoint_info[col]

                funcs = get_well_objective(
                    well_id=info['well_id'],
                    rate_id=info['phase'],
                    step_id=info['steps'],
                    rate=info['rate'],
                    accumulative=info['accumulative'],
                    jl_import=jl
                )

                # Define objective function
                funcs = funcs if isinstance(funcs, list) else [funcs]
                grads = []
                for func in funcs:
                    # Compute adjoint gradient
                    grad = jl.JutulDarcy.reservoir_sensitivities(
                        case, 
                        jlres, 
                        func,
                        include_parameters=True,
                    )
                    grads.append(grad)

                # Extract and store gradients in adjoint dataframe
                for g, grad in enumerate(grads):
                    for param in info['parameters']:
                        index = self.true_order[1][info['steps'][g]-1]
                        
                        if param.lower() == 'poro':
                            grad_param = np.array(grad[jl.Symbol("porosity")])
                            grad_param = _expand_to_active_grid(grad_param, actnum_vec, fill_value=0)
                            adjoints.at[index, (col, param)] = grad_param

                        elif param.lower().startswith('perm'):
                            grad_param = np.array(grad[jl.Symbol("permeability")])

                            m2_per_mD = 9.869233e-16
                            if param.lower() == 'permx':
                                grad_param = grad_param[0] * m2_per_mD
                            elif param.lower() == 'permy':
                                grad_param = grad_param[1] * m2_per_mD
                            elif param.lower() == 'permz':
                                grad_param = grad_param[2] * m2_per_mD
                            
                            grad_param = _expand_to_active_grid(grad_param, actnum_vec, fill_value=0)
                            adjoints.at[index, (col, param)] = grad_param
                        else:
                            raise ValueError(f'Param: {param} not supported for adjoint sensitivity')
                
                
                # Update progress bar
                pbar.update(1)
            pbar.close()

        os.chdir('..')

        # Delete simulation folder
        if delete_folder:
            shutil.rmtree(folder)

        # Extract requested datatypes
        output = self.extract_datatypes(pyres, jlcase=case, out_format=self.out_format)
        
        if self.compute_adjoints:
            return output, adjoints
        else:
            return output

    def render_makofile(self, makofile: str, folder: str, input: dict):
        '''
        Render makofile.mako to makofile.DATA using input
        '''
        self.datafile = makofile.replace('.mako', '.DATA')
        template = Template(filename=makofile)
        with open(os.path.join(folder, self.datafile), 'w') as f:
            f.write(template.render(**input))


    def extract_datatypes(self, res: dict, jlcase =None, out_format='list') -> Union[dict|list|pd.DataFrame]:
        '''
        Extract requested datatypes from simulation results.

        Parameters
        ----------
        res : dict
            Simulation results dictionary.
        out_format : str, optional
            Output format ('list[dict]', 'dict', 'dataframe'), by default 'list'.
        
        Returns
        -------
            Union[dict, list, pd.DataFrame]
                Extracted data in the specified format.
        '''
        out = {}
        for orginal_key in self.datatype:
            key = orginal_key.upper()

            # Check if key is FIELD data
            if key in res['FIELD']:
                out[orginal_key] = res['FIELD'][key]
            # Check if key is WELLS data (format: "DATA:WELL" or "DATA WELL")
            elif ':' in key or ' ' in key:
                data_id, well_id = key.replace(':', ' ').split(' ')
                out[orginal_key] = res['WELLS'][well_id][data_id]

            elif key in [str(k) for k in jlcase.input_data["GRID"].keys()]:
                value = jlcase.input_data["GRID"][f"{key}"]
                try:
                    out[orginal_key] = np.array(value)
                except:
                    out[orginal_key] = value
            else:
                raise KeyError(f'Data type {key} not found in simulation results')
        
        # Format output
        if out_format == 'list':
            # Make into a list of dicts where each dict is a time step (pred_data format)
            out_list = []
            for i in range(len(res['DAYS'])):
                time_step_data = {key: np.array([out[key][i]]) for key in out}
                out_list.append(time_step_data)
            return out_list
        
        elif out_format == 'dict':
            out['DAYS'] = res['DAYS']
            return out

        elif out_format == 'dataframe':
            df = pd.DataFrame(data=out, index=res['DAYS'])
            return df
        
    
def _symdict_to_pydict(symdict, jl_import):
    '''Convert a Julia symbolic dictionary to a Python dictionary recursively.'''
    pydict = {}
    for key, value in symdict.items():
        if jl_import.isa(value, jl_import.AbstractDict):
            pydict[str(key)] = _symdict_to_pydict(value, jl_import)
        else:
            pydict[str(key)] = value
    return pydict
    
def _expand_to_active_grid(param, active, fill_value=np.nan):

    if len(param) == active.sum():
        val = []
        i = 0
        for cell in active:
            if cell == 1:
                val.append(param[i])
                i += 1
            else:
                val.append(fill_value)
    elif len(param) == len(active):
        val = param
    else:
        raise ValueError('Parameter length does not match number of active cells')
    
    return np.array(val)


def get_well_objective(well_id, rate_id, step_id, rate=True, accumulative=True, jl_import=None):
    '''
    Create a Julia objective function for well-based adjoint sensitivity analysis.

    This function generates JutulDarcy objective functions that compute well quantities
    of interest (QOI) for specific phases. The objective can target all timesteps,
    specific timesteps, or a single timestep, and can return either instantaneous
    rates or cumulative volumes.

    Parameters
    ----------
    well_id : str
        Identifier of the well for which to compute the objective.
    rate_id : str
        Phase type identifier. Supported values:
        - 'mass': Total surface mass rate
        - 'liquid': Surface liquid rate
        - 'water': Surface water rate
        - 'oil': Surface oil rate
        - 'gas': Surface gas rate
        - 'rate': Total volumetric rate
    step_id : int, list, np.ndarray, or None
        Timestep specification:
        - None: Compute objective for all timesteps (cumulative)
        - int: Compute objective for a single specific timestep
        - list/array: Compute objectives for multiple specific timesteps
    rate : bool, optional
        If True (default), returns instantaneous rate at timestep(s).
        If False, multiplies rate by dt for cumulative volume contribution.
    jl_import : module, optional
        Julia Main module from juliacall. If None, will import automatically.

    Returns
    -------
    function or list of functions
        - Single Julia objective function if step_id is None or int
        - List of Julia objective functions if step_id is a list/array

    Raises
    ------
    ValueError
        If rate_id is not one of the supported phase types.

    Examples
    --------
    >>> obj = get_well_objective('PROD1', 'oil', None, rate=False)
    >>> obj = get_well_objective('INJ1', 'water', 10, rate=True)
    >>> objs = get_well_objective('PROD2', 'gas', [5, 10, 15], rate=True)
    '''

    if jl_import is None:
        from juliacall import Main as jl_import
        jl_import.seval('using JutulDarcy')

    rate_id_map = {
        'mass': 'TotalSurfaceMassRate',
        'liquid': 'SurfaceLiquidRateTarget',
        'water': 'SurfaceWaterRateTarget',
        'oil': 'SurfaceOilRateTarget',
        'gas': 'SurfaceGasRateTarget',
        'rate': 'TotalRateTarget'
    }
    if rate_id not in rate_id_map:
        raise ValueError(f'Unknown rate type: {rate_id}')
    rate_id = rate_id_map[rate_id]

    if rate:
        dt = ''
    else:
        dt = 'dt*'

    # Case 1: Sum of all timesteps
    #-----------------------------------------------------------------------------
    if accumulative:
        jl_import.seval(f"""
        function objective_function(model, state, dt, step_i, forces)
            rate = JutulDarcy.compute_well_qoi(
                model, 
                state, 
                forces, 
                Symbol("{well_id}"), 
                {rate_id}
            )
            return {dt}rate
        end
        """)
        return jl_import.objective_function
    #-----------------------------------------------------------------------------
    
    # Case 2: Multiple timesteps
    #-----------------------------------------------------------------------------
    elif isinstance(step_id, (list, np.ndarray)):
        objective_steps = []
        for sid in step_id:
            jl_import.seval(f"""
            function objective_function_{sid}(model, state, dt, step_i, forces)
                if step_i != {sid}
                    return 0.0
                else
                    rate = JutulDarcy.compute_well_qoi(
                        model, 
                        state, 
                        forces, 
                        Symbol("{well_id}"), 
                        {rate_id}
                    )
                    return {dt}rate
                end
            end
            """)
            objective_steps.append(jl_import.seval(f'objective_function_{sid}'))
        return objective_steps
    #-----------------------------------------------------------------------------

    # Case 3: Single timestep
    #-----------------------------------------------------------------------------
    else:
        jl_import.seval(f"""
        function objective_function(model, state, dt, step_i, forces)
            if step_i != {step_id}
                return 0.0
            else
                rate = JutulDarcy.compute_well_qoi(
                    model, 
                    state, 
                    forces, 
                    Symbol("{well_id}"), 
                    {rate_id}
                )
                return {dt}rate
            end
        end
        """)
        return jl_import.objective_function
    #-----------------------------------------------------------------------------
