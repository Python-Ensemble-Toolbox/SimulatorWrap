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
import datetime as dt

from mako.template import Template
from typing import Union
from p_tqdm import p_map
from tqdm import tqdm
import io
from contextlib import redirect_stderr, redirect_stdout
#────────────────────────────────────────────────────


__author__ = 'Mathias Methlie Nilsen'
__all__ = ['JutulDarcyWrapper']


#────────────────────────────────────────────────────────────────────────────────────
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
os.environ['PYTHON_JULIACALL_THREADS'] = '1'
os.environ['PYTHON_JULIACALL_OPTLEVEL'] = '3'
warnings.filterwarnings('ignore', message='.*juliacall module already imported.*')
#────────────────────────────────────────────────────────────────────────────────────


PBAR_OPTS = {
    'ncols': 110,
    'colour': "#285475",
    'bar_format': '{desc}: {percentage:3.0f}% [{bar}] {n_fmt}/{total_fmt} │ ⏱ {elapsed}<{remaining} │ {rate_fmt}',
    'ascii': '-◼', # Custom bar characters for a sleeker look
}


class JutulDarcyWrapper:

    def __init__(self, options):
        """
        Initialize the JutulDarcy simulator wrapper.

        This wrapper provides an interface for running JutulDarcy reservoir simulations
        with support for ensemble-based workflows, parallel execution, and automatic
        adjoint sensitivity computation.

        Parameters
        ----------
        options : dict
            Configuration options for the wrapper. The following keys are supported:

            - makofile : str
                Path to the Mako template file (.mako). Required unless 'runfile' is provided.
            - runfile : str
                Path to the runfile (.DATA). If provided, the corresponding .mako file
                is derived. Required unless 'makofile' is provided.
            - reporttype : str, optional
                Type of report index ('days' or 'date'). Default is 'days'.
            - reportpoint : list
                Collection of report points (timesteps or dates). Required.
            - out_format : {'list', 'dict', 'dataframe'}, optional
                Output format for simulation results. Default is 'list'.
            - datatype : list of str, optional
                List of data types to extract from simulation results.
                Default is ['FOPT', 'FGPT', 'FWPT', 'FWIT'].
            - parallel : int, optional
                Number of parallel simulations to run. Default is 1.
            - units : {'metric', 'si', 'field'}, optional
                Unit system for the simulator. Default is 'metric'.
            - adjoints : dict, optional
                Dictionary specifying adjoint sensitivity configuration. When provided,
                adjoint calculations are performed. Structure:
                {datatype: {
                    'well_id': str or list,
                    'parameters': str or list,
                    'steps': 'acc' | 'all' | int
                    }
                }

        Raises
        ------
        ValueError
            If neither 'makofile' nor 'runfile' is provided in options.

        References
        ----------
        [1] Møyner, O. (2025).
            JutulDarcy.jl - a fully differentiable high-performance reservoir simulator
            based on automatic differentiation. Computational Geosciences, 29, Article 30.
            https://doi.org/10.1007/s10596-025-10366-6
        """
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
        self.units      = options.get('units', 'metric') # Not currently used!
        self.datafile = None
        self.compute_adjoints = False

        # Process datatype entries for well data
        datatype = []
        for dt in self.datatype:
            dt = dt.split(' - ')
            if len(dt) == 1:
                datatype.append(dt[0])
            else:
                t = dt[0]
                d = []
                for item in dt[1:]:
                    d.append(f'{t}:{item}')
                datatype.extend(d)
        self.datatype = datatype

        # This is for PET to work properly
        self.input_dict = options
        self.true_order = [self.reporttype, options['reportpoint']]
        self.steps = [i for i in range(len(self.true_order[1]))]

        # Extract adjoint options
        #---------------------------------------------------------------------------------------------------------
        if 'adjoints' in options:
            self.compute_adjoints = True

            self.adjoint_info = {}
            for datatype in options['adjoints']:
                
                # Determine phase and if rate or volumes
                rate_map = {
                    'WOPT': ('oil', True),
                    'WGPT': ('gas', True),
                    'WWPT': ('water', True),
                    'WLPT': ('liquid', True),
                    'WOPR': ('oil', False),
                    'WGPR': ('gas', False),
                    'WWPR': ('water', False),
                    'WLPR': ('liquid', False),
                }
                phase, rate = rate_map[datatype]

                # Determine steps
                steps = options['adjoints'][datatype].get('steps', 'acc')
                accumulative= False

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

                # Store adjoint info for adjoint computations
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

    def __call__(self, inputs: list[dict]):
        """
        Execute parallel forward simulations for all ensemble members.

        This method runs the forward simulation for all input parameter sets in parallel,
        optionally computing adjoint sensitivities if configured.

        Parameters
        ----------
        inputs : list of dict
            List containing input parameter sets, indexed by ensemble member

            number (0, 1, 2, ...). Each value is a dict of parameters for that member.

        Returns
        -------
        output : list of (dict, list, or pd.DataFrame)
            Forward simulation results for all ensemble members, formatted according
            to the 'out_format' option specified during initialization.
        adjoints : pd.DataFrame, optional
            Adjoint sensitivities for all objectives and parameters. Only returned if
            'adjoints' configuration was provided during initialization. Contains
            multi-indexed columns (objective, parameter) and index based on reporttype.

        Notes
        -----
        - Existing simulation folders (En_*) are deleted before execution to prevent
          conflicts from previous runs.
        - Parallel execution uses the number of CPUs specified in the 'parallel' option.
        - Progress is displayed via a progress bar during execution.

        Examples
        --------
        >>> simulator = JutulDarcyWrapper(options)
        >>> inputs = [{'param1': [1, 2, 3]}, {'param1': [1.1, 2.1, 3.1]}]
        >>> results = simulator(inputs)
        """

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
            desc='Simulations',
            leave=False,
            **PBAR_OPTS
        )

        if self.compute_adjoints:
            results, adjoints = zip(*outputs)
            if len(inputs) == 1:
                results  = results[0]
                adjoints = adjoints[0]
            return results, adjoints
        else:
            return outputs
                     

    def run_fwd_sim(self, input: dict, idn: int=0, delete_folder: bool=True):
        """
        Execute a forward reservoir simulation for a single ensemble member.

        This method performs the complete simulation workflow including case setup,
        execution, results extraction, and optional adjoint sensitivity computation.
        The simulation runs in an isolated folder that is optionally cleaned up
        upon completion.

        Parameters
        ----------
        input : dict
            Dictionary containing input parameters for the simulation. Typically includes
            property grids (PERMX, PERMY, PERMZ, PORO) and other reservoir model parameters.
        idn : int, optional
            Ensemble member identifier (0, 1, 2, ...). Used to name the simulation folder
            and track adjoint results. Default is 0.
        delete_folder : bool, optional
            If True, the simulation folder (En_{idn}) is deleted after completion.
            If False, the folder and all output files are retained. Default is True.

        Returns
        -------
        output : dict, list, or pd.DataFrame
            Forward simulation results in the format specified by the 'out_format' option:
            - 'list': List of dictionaries, one per report point
            - 'dict': Dictionary with results as lists
            - 'dataframe': pandas DataFrame with results
        adjoints : pd.DataFrame, optional
            Adjoint sensitivities with multi-indexed columns (objective, parameter).
            Only returned if 'adjoints' configuration was provided during initialization.

        Notes
        -----
        - A dedicated folder En_{idn} is created for each simulation and contains
          the rendered .DATA file and all JutulDarcy output.
        - Julia output and warnings are suppressed during case setup unless suppress
          flag is explicitly disabled in the code.
        - Adjoint computation includes unit conversions (e.g., mD for permeability,
          Sm3 for volumes) and handles both active and inactive grid cells.

        Raises
        ------
        FileNotFoundError
            If the datafile cannot be found after rendering the makofile.
        KeyError
            If required simulation results are not found in JutulDarcy output.
        """
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

        # Setup case from datafile (suppress all output including well processing messages)
        suppress = True
        if suppress:
            case = jl.seval(f"""
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    setup_case_from_data_file(
                        "{self.datafile}";
                        parse_arg=(silent=true, verbose=false, warn_parsing=false, warn_feature=false)
                    )
                end
            end
            """)
        else:
            case = jl.setup_case_from_data_file(self.datafile)
            

        # Get Units
        if jl.haskey(case.input_data["RUNSPEC"], "METRIC"):
            units = 'metric'
        elif jl.haskey(case.input_data["RUNSPEC"], "SI"):
            units = 'si'
        elif jl.haskey(case.input_data["RUNSPEC"], "FIELD"):
            units = 'field'
        else:
            units = jl.missing

        # Get some grid info
        nx, ny, nz = case.input_data["GRID"]["cartDims"]
        grid = (nx, ny, nz)
        try:
            actnum = np.array(case.input_data["GRID"]["ACTNUM"]) # Shape (nx, ny, nz)
            actnum_vec = actnum.flatten(order='F')  # Fortran order flattening
        except:
            actnum_vec = np.ones(nx*ny*nz)

        # Simulate and get results
        jlres = jl.simulate_reservoir(case, info_level=-1)
        pyres = convert_to_pydict(jlres, case, units=units)
        pyres = self.results_to_dataframe(pyres, self.datatype, jlcase=case, jl_import=jl)

        # Convert output to requested format
        if not self.out_format == 'dataframe':
            if self.out_format == 'dict':
                output = pyres.to_dict(orient='list')
            elif self.out_format == 'list':
                output = []
                for idx, row in pyres.iterrows():
                    row_dict = row.to_dict()
                    row_dict = {k: np.atleast_1d(v) for k, v in row_dict.items()}
                    output.append(row_dict)
        else:
            output = pyres

        
        # Compute adjoints
        if self.compute_adjoints:

            # Initialize adjoint dataframe
            colnames = []
            for key in self.adjoint_info:
                for param in self.adjoint_info[key]['parameters']:
                    colnames.append((key, param))

            adjoints = pd.DataFrame(columns=pd.MultiIndex.from_tuples(colnames), index=self.true_order[1])
            adjoints.index.name = self.true_order[0]
            attrs = {}

            # Initialize progress bar
            PBAR_OPTS.pop('colour', None)
            pbar = tqdm(
                adjoints.keys(), 
                desc=f'Solving adjoints for En_{idn}',
                position=idn+1,
                leave=False,
                unit='obj',
                dynamic_ncols=False,
                colour="#713996",
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
                jl.case = case
                jl.jlres = jlres
                for func in funcs:
                    # Compute adjoint gradient
                    jl.func = func
                    grad = jl.seval("""
                    redirect_stdout(devnull) do
                        redirect_stderr(devnull) do
                            JutulDarcy.reservoir_sensitivities(
                                case, 
                                jlres, 
                                func,
                                include_parameters=true
                            )
                        end
                    end
                    """)
                    grads.append(grad)

                # Extract and store gradients in adjoint dataframe
                for g, grad in enumerate(grads):
                    for param in info['parameters']:
                        index = self.true_order[1][info['steps'][g]]
                        
                        if param.lower() == 'poro':
                            grad_param = np.array(grad[jl.Symbol("porosity")])
                            grad_param = _expand_to_active_grid(grad_param, actnum_vec, fill_value=0)
                            adjoints.at[index, (col, param)] = grad_param
                            attrs[(col, param)] = {'unit': 'Sm3'}

                        elif 'perm' in param.lower():
                            grad_param = np.array(grad[jl.Symbol("permeability")])
                            mD_per_m2 = _convert_from_si(1.0, 'darcy', jl) * 1e3
                            grad_param  = grad_param/mD_per_m2  # Convert from m2 to mD

                            if info['rate']:
                                days_per_sec = _convert_from_si(1.0, 'day', jl)
                                grad_param = grad_param/days_per_sec
                                attrs[(col, param)] = {'unit': 'Sm3/(day∙mD)'}
                            else:
                                attrs[(col, param)] = {'unit': 'Sm3/mD'}

                            if param.lower() == 'permx':
                                grad_param = grad_param[0]
                            elif param.lower() == 'log_permx':
                                permx = np.array(case.input_data["GRID"]["PERMX"])
                                permx = _convert_from_si(permx, 'darcy', jl_import=jl)
                                permx = np.array(permx)* 1e3 # Darcy to mD
                                grad_param = grad_param[0]/permx.flatten(order='F')

                            elif param.lower() == 'permy':
                                grad_param = grad_param[1]
                            elif param.lower() == 'log_permy':
                                permy = np.array(case.input_data["GRID"]["PERMY"])
                                permy = _convert_from_si(permy, 'darcy', jl_import=jl)
                                permy = np.array(permy)* 1e3 # Darcy to mD
                                grad_param = grad_param[1]/permy.flatten(order='F')

                            elif param.lower() == 'permz':
                                grad_param = grad_param[2]
                            elif param.lower() == 'log_permz':
                                permz = np.array(case.input_data["GRID"]["PERMZ"])
                                permz = _convert_from_si(permz, 'darcy', jl_import=jl)
                                permz = np.array(permz)* 1e3 # Darcy to mD
                                grad_param = grad_param[2]/permz.flatten(order='F')
                            
                            grad_param = _expand_to_active_grid(grad_param, actnum_vec, fill_value=0)
                            adjoints.at[index, (col, param)] = grad_param

                        else:
                            raise ValueError(f'Param: {param} not supported for adjoint sensitivity')
                
                
                # Update progress bar
                pbar.update(1)
            pbar.close()
            adjoints.attrs = attrs

        os.chdir('..')
        

        # Delete simulation folder
        if delete_folder:
            shutil.rmtree(folder)
        
        if self.compute_adjoints:
            return output, adjoints
        else:
            return output

    def render_makofile(self, makofile: str, folder: str, input: dict):
        """
        Render Mako template file to Eclipse-format DATA file.

        This method uses the Mako templating engine to render a .mako template
        file into an Eclipse-format .DATA file, substituting template variables
        with values from the input dictionary.

        Parameters
        ----------
        makofile : str
            Path to the Mako template file (.mako extension).
        folder : str
            Directory where the rendered .DATA file will be written.
        input : dict
            Dictionary of template variables to be substituted in the Mako template.
            Keys should match template variable names.

        Returns
        -------
        None

        Side Effects
        ____________
        - Creates a .DATA file in the specified folder with the same basename as makofile.
        - Updates self.datafile with the path to the rendered DATA file.

        Raises
        ------
        FileNotFoundError
            If the makofile does not exist.
        MakoException
            If the Mako template contains syntax errors or undefined variables.
        """
        self.datafile = makofile.replace('.mako', '.DATA')
        template = Template(filename=makofile)
        with open(os.path.join(folder, self.datafile), 'w') as f:
            f.write(template.render(**input))


    def results_to_dataframe(self, res: dict, datatypes: list, jlcase=None, jl_import=None) -> pd.DataFrame:
        """
        Convert simulation results to a pandas DataFrame with metadata.

        This method extracts specified datatypes from JutulDarcy simulation results
        and organizes them into a structured DataFrame with proper units and indexing.
        Supports both temporal data (from FIELD and WELLS) and static grid properties.

        Parameters
        ----------
        res : dict
            Simulation results dictionary from JutulDarcy containing keys:
            - 'FIELD': Dictionary of field-level data (FOPT, FGPT, etc.)
            - 'WELLS': Dictionary of well-level data by well name
            - 'DAYS': Array of timestep values in days
        datatypes : list of str
            Data types to extract. Formats supported:
            - Field data: 'FOPT', 'FGPT', 'FOPR', etc.
            - Well data: 'WOPT:WELL1', 'WWPR:INJ1' (colon or space separated)
            - Grid properties: 'PERMX', 'PERMY', 'PERMZ', 'PORO'
        jlcase : JutulDarcy.SimulationCase, optional
            JutulDarcy simulation case object containing grid and input data.
            Required for accessing grid properties (PERMX, PERMY, etc.).
        jl_import : module, optional
            Julia Main module from juliacall. Used for unit conversions.
            Required if datatypes include permeability values.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by reportpoint (days or dates) with:
            - Columns: Requested datatypes
            - Attribute 'attrs': Dictionary mapping datatype to metadata (unit information)
            - Index name: 'days' or 'date' based on reporttype configuration

        Notes
        -----
        - Permeability values are converted from SI units (m²) to millidarcies (mD).
        - For well data, if the well is not found, that (time, datatype) entry remains NaN.
        - Grid properties are typically singleton values assigned to the first index.
        - Date conversion uses the START date from RUNSPEC section.

        Raises
        ------
        KeyError
            If a requested datatype is not found in field data, well data, or grid properties.
        ValueError
            If datatypes are malformed or reference non-existent wells.
        """
        
        # Get start date
        start_date = jlcase.input_data["RUNSPEC"]['START']

        df = pd.DataFrame(columns=datatypes, index=self.true_order[1])
        df.index.name = self.true_order[0]
        attrs = {}

        for key in datatypes:
            key_upper = key.upper()

            # Check if key is FIELD data
            if key_upper in res['FIELD']:
                for d, day in enumerate(res['DAYS']):
                    if self.true_order[0] == 'days':
                        if day in self.true_order[1]:
                            df.at[day, key] = res['FIELD'][key_upper][d]
                    elif self.true_order[0] == 'date':
                        date = start_date + dt.timedelta(days=int(day))
                        if date in self.true_order[1]:
                            df.at[date, key] = res['FIELD'][key_upper][d]
                attrs[key_upper] = {'unit': _metric_unit(key_upper)}

            # Check if key is WELLS data (format: "DATA:WELL" or "DATA WELL")
            elif ':' in key_upper or ' ' in key_upper:
                datakey, wid = key_upper.replace(':', ' ').split(' ')
                for d, day in enumerate(res['DAYS']):
                    
                    if self.true_order[0] == 'days':
                        if day in self.true_order[1]:
                            df.at[day, key] = res['WELLS'][wid][datakey][d]
                    elif self.true_order[0] == 'date':

                        date = start_date + dt.timedelta(days=int(day))
                        if date in self.true_order[1]:
                            df.at[date, key] = res['WELLS'][wid][datakey][d]

                attrs[key_upper] = {'unit': _metric_unit(key.upper())}

            elif key_upper in [str(k) for k in jlcase.input_data["GRID"].keys()] and (jlcase is not None):
                value = jlcase.input_data["GRID"][f"{key_upper}"]
                
                if key_upper.startswith('PERM'):
                    value = _convert_from_si(value, 'darcy', jl_import)
                    value = np.array(value) * 1e3 # Darcy to mD
                    attrs[key_upper] = {'unit': 'mD'}
                else:
                    attrs[key_upper] = {'unit': _metric_unit(key_upper)}

                try:
                    df.at[df.index[0], key] = np.array(value)
                except:
                    df.at[df.index[0], key] = value
                
            else:
                raise KeyError(f'Data type {key} not found in simulation results')
            
        df.attrs = attrs
        return df

def _extract_well_data(wdata, datakey, wid, reporttype, reportpoint):
    pass

def _extract_field_data(fdata, datakey, reporttype, reportpoint):
    pass

def _extract_grid_property(gdata, datakey):
    pass
    

def _symdict_to_pydict(symdict, jl_import):
    """
    Recursively convert Julia symbolic dictionary to Python dictionary.

    This utility function converts JutulDarcy result dictionaries containing Julia
    symbolic keys and nested dictionaries to native Python types.

    Parameters
    ----------
    symdict : Julia AbstractDict
        Julia dictionary object with symbolic keys.
    jl_import : module
        Julia Main module from juliacall for type checking.

    Returns
    -------
    dict
        Python dictionary with string keys and recursively converted values.
        All Julia AbstractDict values are recursively converted.

    Notes
    -----
    - Recursion handles arbitrarily nested dictionary structures.
    - Julia symbols are converted to Python strings.
    - Non-dictionary values are returned unchanged (caller is responsible for
      converting array and numeric types as needed).
    """
    pydict = {}
    for key, value in symdict.items():
        if jl_import.isa(value, jl_import.AbstractDict):
            pydict[str(key)] = _symdict_to_pydict(value, jl_import)
        else:
            pydict[str(key)] = value
    return pydict
    
def _expand_to_active_grid(param, active, fill_value=np.nan):
    """
    Expand parameter values from active cells to full grid.

    This utility function maps values defined only for active cells to correspond-
    ing positions in the full grid (including inactive cells), which is necessary
    because JutulDarcy computations often only involve active grid cells.

    Parameters
    ----------
    param : array-like
        Parameter values, either for active cells only (length = active.sum())
        or for all cells (length = len(active)).
    active : array-like
        Boolean or binary (0/1) array indicating active cells. Shape (n_cells,).
    fill_value : float, optional
        Value to assign to inactive cells. Default is np.nan.

    Returns
    -------
    np.ndarray
        Array of length len(active) with values expanded to full grid:
        - Active cell positions: values from param
        - Inactive cell positions: fill_value

    Raises
    ------
    ValueError
        If param length is neither active.sum() nor len(active).

    Notes
    -----
    - Fortran-order (column-major) cell indexing is assumed for mapping.
    - This is typically used for gradient expansion in sensitivity analysis.
    """
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


def _convert_from_si(value, unit, jl_import):
    """
    Convert numerical values from SI units to specified unit system.

    This utility function leverages JutulDarcy's built-in unit conversion
    to transform SI-based values to alternative unit systems (e.g., millidarcies,
    days, surface cubic meters).

    Parameters
    ----------
    value : float or array-like
        Numerical value(s) in SI units to convert.
    unit : str
        Target unit name as string. Common values: 'darcy', 'day', 'Sm3'.
        Full list of supported units depends on Jutul.UnitSystem.
    jl_import : module
        Julia Main module from juliacall providing Jutul.convert_from_si.

    Returns
    -------
    float or array-like
        Converted value(s) in specified units, matching input type.

    Notes
    -----
    - Permeability: SI is m², commonly converted to mD (millidarcies) using factor ≈ 1e15.
    - Time: SI is seconds, commonly converted to days using factor ≈ 1.157e-5.
    - This is a thin wrapper around Jutul.convert_from_si for convenience.
    """
    return jl_import.Jutul.convert_from_si(value, jl_import.Symbol(unit))

def _metric_unit(key: str) -> str:
    """
    Return the metric unit string for a given data key.

    Maps Eclipse output keywords and property names to their corresponding
    metric/SI units used in the JutulDarcy simulator.

    Parameters
    ----------
    key : str
        Eclipse keyword or property name. Case-insensitive.
        Examples: 'FOPT', 'FOPR', 'WOPR', 'PERMX', 'PORO', 'WWIR'.

    Returns
    -------
    str
        Unit string corresponding to the key. Returns 'Unknown' if key is not
        in the predefined mapping.

    Notes
    -----
    Supported keywords and their units:
    - PORO: dimensionless fraction
    - PERM*: millidarcies (mD)
    - FOPT, FGPT, FWPT, FWLT, FWIT: surface cubic meters (Sm3)
    - FOPR, FGPR, FWPR, FLPR, FWIR: volumetric flow (Sm3/day)
    - WOPR, WGPR, WWPR, WLPR, WWIR: well volumetric flow (Sm3/day)

    Examples
    --------
    >>> _metric_unit('FOPT')
    'Sm3'
    >>> _metric_unit('PERMX')
    'mD'
    >>> _metric_unit('PORO')
    ''
    """
    unit_map = {
        'PORO': '',
        'PERMX': 'mD',
        'PERMY': 'mD',
        'PERMZ': 'mD',
        #---------------------
        'FOPT': 'Sm3',
        'FGPT': 'Sm3',
        'FWPT': 'Sm3',
        'FWLT': 'Sm3',
        'FWIT': 'Sm3',
        #---------------------
        'FOPR': 'Sm3/day',
        'FGPR': 'Sm3/day',
        'FWPR': 'Sm3/day',
        'FLPR': 'Sm3/day',
        'FWIR': 'Sm3/day',
        #---------------------
        'WOPR': 'Sm3/day',
        'WGPR': 'Sm3/day',
        'WWPR': 'Sm3/day',
        'WLPR': 'Sm3/day',
        'WWIR': 'Sm3/day',
    }
    if key.upper() in unit_map:
        return unit_map[key.upper()]
    else:
        return 'Unknown'


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
                if step_i != {sid+1}
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
            if step_i != {step_id+1}
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
