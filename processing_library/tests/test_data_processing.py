# tests/test_data_processing.py

import pytest
import pandas as pd
from battery_analysis.data_processing import psuedo_limit_extraction, add_cv_steps

def test_psuedo_limit_extraction():
    # Prepare minimal test data
    df = pd.DataFrame({
        'time': [0, 1, 2, 3, 4, 120, 121],
        'current': [0, 1, 1, 1, 0, -1, -1],
        'voltage': [3.2, 3.3, 3.35, 3.4, 3.45, 3.4, 3.3]
    })
    df_processed, pseudo_high, pseudo_low = psuedo_limit_extraction(df)

    # Assert shape is same
    assert df_processed.shape[0] == 7

    # Ensure step type columns exist
    assert "Step Type" in df_processed.columns

    # pseudo_high and pseudo_low should be floats
    assert isinstance(pseudo_high, float)
    assert isinstance(pseudo_low, float)

def test_add_cv_steps():
    # Prepare minimal test data
    df = pd.DataFrame({
        'time': [0, 1, 2, 3, 4, 5, 6],
        'current': [1.0, 1.0, 0.9, 0.8, 0.05, 0.03, 0],
        'voltage': [3.0, 3.5, 3.6, 3.7, 3.85, 3.9, 3.95],
        'group': [1, 1, 1, 1, 1, 1, 1],
        'Step Type': ['charge', 'charge', 'charge', 'charge', 'charge', 'charge', 'charge']
    })

    df_processed = add_cv_steps(df, vhigh=4.2, vlow=2.5)
    # Check if 'charge cv' is assigned
    assert 'charge cv' in df_processed['Step Type'].unique() or 'discharge cv' in df_processed['Step Type'].unique()