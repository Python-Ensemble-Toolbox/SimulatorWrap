'''
Simulator wrapper for the JutulDarcy simulator.

This module provides a wrapper interface for running JutulDarcy simulations
with support for ensemble-based workflows and flexible output formatting.
'''

#────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import shutil
import os

from mako.template import Template
#────────────────────────────────────────────────────


__author__ = 'Mathias Methlie Nilsen'
__all__ = ['JutulDarcyWrapper']


class JutulDarcyWrapper:

    def __init__(self, options):

        # Make makofile an mandatory option
        if 'makofile' not in options:
            raise ValueError('Wrapper  requires a makofile option')
        else:
            self.makofile = options.get('makofile')

        # Other variables
        self.reporttype = options.get('reporttype', 'days')
        self.out_format = options.get('out_format', 'list')
        self.datatype   = options.get('datatype', ['FOPT', 'FGPT', 'FWPT', 'FWIT'])
        self.parallel   = options.get('parallel', 1)
        self.platform   = options.get('platform', 'Python')
        self.datafile   = None

        # This is for PET to work properly (should be removed in future versions)
        self.input_dict = options
        self.true_order = [self.reporttype, options['reportpoint']]


    def run_fwd_sim(self, input: dict, idn: int=0, delete_folder: bool=True) -> dict:
        '''
        Run forward simulation for given input parameters.

        Parameters
        ----------
        input : dict
            Input parameters for the simulation.

        idn : int, optional
            Ensemble member ID, by default 0.

        delete_folder : bool, optional
            Whether to delete the simulation folder after running, by default True.
        '''

        # Include ensemble member id in input dict
        input['member'] = idn
        
        # Make simulation folder
        folder = f'En_{idn}'
        os.makedirs(folder)

        # Render makofile
        self.render_makofile(self.makofile, folder, input)

        # Enter simulation folder and run simulation
        os.chdir(folder)

        if self.platform == 'Python':
            # Needs to be imported here for multiprocessing to work
            from jutuldarcy import simulate_data_file
            res = simulate_data_file(
                data_file_name=self.datafile, 
                convert=True, # Convert to output dictionary
                units='si',   # Use SI units (Sm3 and so on)
                info_level=-1 # No terminal output
            )
        elif self.platform == 'Julia':
            from jutuldarcy import convert_to_pydict
            from juliacall import Main as jl
            jl.seval("using JutulDarcy, Jutul")
            case  = jl.setup_case_from_data_file(self.datafile)
            jlres = jl.simulate_reservoir(case, info_level=-1) 
            res   = convert_to_pydict(jlres, case, units='si')

            # TODO: Make sure the gradient computation works (this example is hardcoded, for a specific case)
            if False:
                # Define objective function for gas rate
                jl.seval("""
                function objective_function(model, state, dt, step_i, forces)
                    oil_rate = JutulDarcy.compute_well_qoi(model, state, forces, :PROD, SurfaceOilRateTarget)
                    return dt*oil_rate
                end
                """)
                
                # Compute sensitivities with respect to parameters
                sensitivities = jl.JutulDarcy.reservoir_sensitivities(
                    case, 
                    jlres, 
                    jl.objective_function,
                    include_parameters=True
                )
                
                # Access permeability gradient
                poro_grad = sensitivities[jl.Symbol("porosity")]
                # Convert to numpy array
                poro_gradient = np.array(poro_grad)
                
        os.chdir('..')

        # Delete simulation folder
        if delete_folder:
            shutil.rmtree(folder)

        # Extract requested datatypes
        output = self.extract_datatypes(res, out_format=self.out_format)
        return output


    def render_makofile(self, makofile: str, folder: str, input: dict):
        '''
        Render makofile.mako to makofile.DATA using input
        '''
        self.datafile = makofile.replace('.mako', '.DATA')
        template = Template(filename=makofile)
        with open(os.path.join(folder, self.datafile), 'w') as f:
            f.write(template.render(**input))


    def extract_datatypes(self, res: dict, out_format='list') -> dict:
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