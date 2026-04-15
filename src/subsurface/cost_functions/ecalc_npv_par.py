import pandas as pd
import numpy  as np
import os
import yaml
import time

from pathlib import Path
from multiprocessing import Pool

HERE = Path().cwd()  # fallback for ipynb's
HERE = HERE.resolve()

def ecalc_npv_par(pred_data, **kwargs):
    '''
    Parameters
    ----------
    pred_data : array_like
        Ensemble of predicted data.

    **kwargs : dict
        Other arguments sent to the npv function

        keys_opt : list
            Keys with economic data.

        report : list
            Report dates.

    Returns
    -------
    objective_values : array_like
        Objective function values (NPV) for all ensemble members.
    '''
    global const
    global report_dates
    global sim_kwargs
    global sim_data

    # Get the necessary input
    keys_opt = kwargs.get('input_dict', {})
    report = kwargs.get('true_order', [])

    # Economic and other constatns
    const = dict(keys_opt['npv_const'])
    sim_kwargs = keys_opt
    report_dates = report[1]

    # Handle levels
    nt = len(pred_data) # number of report-dates (nt)
    L = None
    for t in np.arange(nt):
        if not isinstance(pred_data[t], list):
            pred_data[t] = [pred_data[t]]
    L = len(pred_data[0])

    # define a data getter
    get_data = lambda i, l, key: pred_data[i + 1][l][key.upper()].squeeze() - pred_data[i][l][key.upper()].squeeze()

    # ensemble size (ne)
    try:
        ne = len(get_data(1, 0, 'fopt'))
    except:
        ne = 1

    # Loop over levels
    objective = []
    for l in range(L):

        objective.append([])
        sim_data = {'fopt': np.zeros((ne, nt-1)),
                    'fgpt': np.zeros((ne, nt-1)),
                    'fwpt': np.zeros((ne, nt-1)),
                    'fwit': np.zeros((ne, nt-1)),
                    'days': np.zeros(nt-1)}

        # loop over pred_data
        for t in range(nt-1):

            for datatype in ['fopt', 'fgpt', 'fwpt', 'fwit']:
                sim_data[datatype][:,t] = get_data(t, l, datatype)

            # days in time-step
            sim_data['days'][t] = (report_dates[t+1] - report_dates[t]).days

        # calculate NPV values
        indecies = [n for n in range(ne)]

        n_par = keys_opt.get('parallel', 50)
        with Pool(n_par) as pool:
            values = pool.map(_objective, indecies)

        objective[l].extend(values)
        objective[l] = np.array(objective[l]) / const.get('obj_scaling', 1)

    return objective

def _objective(args):
    nc = args
    eCalc_data = get_eCalc_data(nc)

    # calc co2 emissions with eCalc
    emissions_total, energy_total = run_eCalc(nc, eCalc_data) # ton/day

    # Total energy usage and emissions over the whole time period (in MWd and tons, respectively)
    el_usage = energy_total.values * sim_data['days']  # total number of MWd (energy usage in MW * number of days)
    em_mass = emissions_total.values * sim_data['days']  # total number of tons

    oil_export = sim_data['fopt'][nc]
    gas_export = sim_data['fgpt'][nc]
    water_prod = sim_data['fwpt'][nc]
    water_inj = sim_data['fwit'][nc]

    value = (oil_export * const['wop'] + gas_export * const['wgp'] - water_prod * const['wwp'] -
             water_inj * const['wwi'] - em_mass * const['wem'] - el_usage * const['wel']) / (
                    (1 + const['disc']) ** (np.cumsum(sim_data['days']) / 365))

    return np.sum(value)

def run_eCalc(n: int, ecalc_data: dict):
    from libecalc.application.energy_calculator import EnergyCalculator
    from libecalc.common.time_utils import Frequency
    from libecalc.presentation.yaml.model import YamlModel
    from ecalc_cli.infrastructure.file_resource_service import FileResourceService
    from libecalc.presentation.yaml.file_configuration_service import FileConfigurationService
    
    pd.DataFrame(ecalc_data).to_csv(f'ecalc_input_{n}.csv', index=False)
    new_yaml = duplicate_yaml_file(sim_kwargs['ecalc_yamlfile'], n)

    # Config
    model_path = HERE/new_yaml
    configuration_service = FileConfigurationService(configuration_path=model_path)
    resource_service = FileResourceService(working_directory=model_path.parent)
    yaml_model = YamlModel(configuration_service=configuration_service,
                           resource_service=resource_service,
                           output_frequency=Frequency.NONE)

    # Compute energy, emissions
    model = EnergyCalculator(graph=yaml_model.get_graph())
    consumer_results = model.evaluate_energy_usage(yaml_model.variables)
    emission_results = model.evaluate_emissions(yaml_model.variables, consumer_results)

    # Extract
    energy = results_as_df(yaml_model, consumer_results, lambda r: r.component_result.energy_usage)
    energy_total = energy.sum(1).rename("energy_total")
    energy_total.to_csv(HERE / "energy.csv")
    emissions = results_as_df(yaml_model, emission_results, lambda r: r['co2_fuel_gas'].rate)
    emissions_total = emissions.sum(1).rename("emissions_total")
    emissions_total.to_csv(HERE / "emissions.csv")
    
    # delete dummy files
    os.remove(new_yaml)
    os.remove(f'ecalc_input_{n}.csv')

    return emissions_total, energy_total

def results_as_df(yaml_model, results, getter) -> pd.DataFrame:
    """Extract relevant values, as well as some meta (`attrs`)."""
    df = {}
    attrs = {}
    res = None
    for id_hash in results:
        res = results[id_hash]
        res = getter(res)
        component = yaml_model.get_graph().get_node(id_hash)
        df[component.name] = res.values
        attrs[component.name] = {'id_hash': id_hash,
                                 'kind': type(component).__name__,
                                 'unit': res.unit}
    if res is None:
        sys.exit('No emission results from eCalc!')
    df = pd.DataFrame(df, index=res.timesteps)
    df.index.name = "dates"
    df.attrs = attrs
    return df


def duplicate_yaml_file(filename, member):

    # Load the YAML file
    try:
        with open(filename, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
    except:
        time.sleep(2)

    input_name = data['TIME_SERIES'][0]['FILE']
    data['TIME_SERIES'][0]['FILE'] = input_name.replace('.csv', f'_{member}.csv')

    # Write the updated content to a new file
    new_filename = filename.replace(".yaml", f"_{member}.yaml")
    with open(new_filename, 'w') as new_yaml_file:
        yaml.dump(data, new_yaml_file, default_flow_style=False)

    return new_filename


def get_eCalc_data(n):

    data = {'dd-mm-yyyy' : [],
            'OIL_PROD'   : [],
            'GAS_PROD'   : [], 
            'WATER_INJ'  : []}

    # Loop over sim_data
    nt = sim_data['fopt'].shape[1] # number of time-steps (nt)
    for t in range(nt):
        D = report_dates[t]
        data['dd-mm-yyyy'].append(D.strftime("%d/%m/%Y"))
        data['OIL_PROD'].append(sim_data['fopt'][n, t] / sim_data['days'][t])
        data['GAS_PROD'].append(sim_data['fgpt'][n, t] / sim_data['days'][t])
        data['WATER_INJ'].append(sim_data['fwit'][n, t] / sim_data['days'][t])
   
    return data
