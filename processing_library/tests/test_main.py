# tests/test_main.py

import pytest
import pandas as pd
from battery_analysis.main import process_lifetime_test

def test_process_lifetime_test_end_to_end():
    # Minimal DataFrame
    df = pd.DataFrame({
        'time (s)': [0, 1, 2, 3],
        'current (a)': [0, 1, -1, 0],
        'voltage (v)': [3.2, 3.3, 3.1, 3.0],
        'charge capacity (ah)': [0, 0.5, 0.5, 0.5],
        'discharge capacity (ah)': [0, 0, 0.4, 0.4],
        'charge energy (wh)': [0, 1.6, 1.6, 1.6],
        'discharge energy (wh)': [0, 0, 1.2, 1.2]
    })

    base_config = {
        "targets": {
            "Rate Performance Test": [
                {
                    "key": "standard_dch_capacity",
                    "group_type": "discharge",
                    "full_cycle": True,
                    "range_c_rate": [0.25, 0.5],
                    "interest_variable": "capacity",
                    "per_cycle": False
                }
            ]
        }
    }

    combined_input = {
        "test_type": "Rate Performance Test",
        "cell_limits": {
            "capacity": 0.5
        },
        "user_input": {
            "pulse_durations": [1, 2]
        }
    }

    result_df = process_lifetime_test(df, combined_input, base_config)
    # The result can vary, but ensure it is not empty
    assert result_df is not None
    assert not result_df.empty