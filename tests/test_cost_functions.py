import numpy as np
import pandas as pd
import pytest
from subsurface.cost_functions import npv

INDEX = pd.date_range("2020-01-01", "2025-01-01", freq="MS")
FOPT = np.linspace(0, 500_000,  len(INDEX))
FGPT = np.linspace(0, 2_000_000, len(INDEX))
FWPT = np.linspace(0, 100_000,  len(INDEX))
FWIT = np.linspace(0, 100_000,  len(INDEX))

def make_df(index=None, fopt=None, fgpt=None, fwpt=None, fwit=None):
    if index is None: index = INDEX
    data = {
        "FOPT": fopt if fopt is not None else FOPT,
        "FGPT": fgpt if fgpt is not None else FGPT,
        "FWPT": fwpt if fwpt is not None else FWPT,
        "FWIT": fwit if fwit is not None else FWIT,
    }
    return pd.DataFrame(data, index=index)

def test_npv_basic_upper_case():
    '''Test that npv returns a reasonable value for typical input data.'''
    data = make_df()
    npv_val = npv(data)
    npv_expected = 190126397.93248096  # Expected NPV based on default parameters
    assert np.isclose(npv_val, npv_expected, rtol=1e-6)

def test_npv_basic_lower_case():
    '''Test that npv works with lower-case column names.'''
    data = make_df()
    data.columns = [col.lower() for col in data.columns]
    npv_val = npv(data)
    npv_expected = 190126397.93248096  # Expected NPV based on default parameters
    assert np.isclose(npv_val, npv_expected, rtol=1e-6)

def test_npv_missing_columns_error():
    '''Test that npv raises an error if required columns are missing.'''
    data = make_df()
    data = data.drop(columns=["FGPT"])
    with pytest.raises(KeyError):
        npv(data)


