"""Net present value."""
import numpy  as np
import pandas as pd
import warnings

DEFAULT_ECON = {
    'wop': 471.0,  # Oil price: $/Sm3 (equivalent to 75 $/STB)
    'wgp': 0.4,    # Gas price: $/Sm3
    'wwp': 40.0,   # Cost of water production: $/Sm3
    'wwi': 25.0,   # Cost of water injection: $/Sm3
    'wem': 150.0,  # Cost of CO2 emissions: $/ton
    'wfu': 80.0,   # Cost of diesel fuel: $/ton
    'disc': 0.08,  # Discount rate per year: 8%
}

__all__ = ['npv']

def npv(data: pd.DataFrame, **kwargs):
    # --- Extract economic parameters and scaling factor ---
    input_dict = kwargs.get("input_dict", {})
    econ_prms = DEFAULT_ECON
    econ_prms.update(input_dict.get("npv_const", {}))
    scaling_factor = econ_prms.pop("obj_scaling", 1.0)
    
    # --- Extract relevant columns from the DataFrame ---
    vol_oil   = _get_column(data, "FOPT").diff()      # Oil production volume (Sm3)
    vol_gas   = _get_column(data, "FGPT").diff()      # Gas production volume (Sm3)
    vol_wpr   = _get_column(data, "FWPT").diff()      # Water production volume (Sm3)
    vol_win   = _get_column(data, "FWIT").diff()      # Water injection volume (Sm3)

    co2_rate  = _get_column(data, "FU_CO2R")          # CO2 emission rate (ton/day)
    fuel_rate = _get_column(data, "FU_FUEL")          # Fuel consumption rate (Sm3/day)
    dt = data.index.to_series().diff().dt.days
    co2_mass  = co2_rate * dt                              # CO2 mass (ton/day)
    fuel_mass = fuel_rate * dt                             # Fuel mass (Sm3/day)

    # --- Compute time in years from start ---
    time_index = data.index.to_numpy()
    years = (time_index - time_index[0]) / np.timedelta64(365, "D")
    
    # --- Revenue components ---
    revenue = (
        vol_oil * econ_prms["wop"] + 
        vol_gas * econ_prms["wgp"]
    )

    # --- Cost components ---
    costs = (
        vol_wpr * econ_prms["wwp"] +
        vol_win * econ_prms["wwi"] +
        co2_mass * econ_prms["wem"] +
        fuel_mass * econ_prms["wfu"]
    )

    # --- Net discounted cash flow ---
    discount_factor = (1 + econ_prms["disc"]) ** years
    discounted_npv = (revenue - costs) / discount_factor
    
    # --- Return scaled NPV ---
    return discounted_npv.sum() / scaling_factor



def _get_column(data: pd.DataFrame, name: str) -> pd.Series:
    colmap = {c.lower(): c for c in data.columns}
    key = name.lower()
    
    if key in colmap:
        col = data[colmap[key]]
        
        # Handle NaN / None
        if col.isnull().any():
            warnings.warn(
                f"Column '{name}' contains None or NaN values. Returning zeros for those entries."
            )
            col = col.fillna(0)
        return col

    warnings.warn(
        f"Column '{name}' not found (case-insensitive). Returning zeros."
    )
    return pd.Series(np.zeros(len(data)), index=data.index)
