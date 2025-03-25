# tests/test_analysis_aggregator.py

import numpy as np
import pandas as pd
import pytest
from battery_analysis.analysis_aggregator import find_parameters_for_section


def test_find_parameters_for_section():
    # minimal group data
    groups = [
        {"group_type": "discharge", "capacity": 2.0, "c-rate": 0.3},
        {"group_type": "charge", "capacity": 2.2, "c-rate": 0.3},
    ]
    # minimal targets
    targets = [
        {"key": "capacity_dch", "group_type": "discharge", "interest_variable": "capacity"},
        {"key": "capacity_ch", "group_type": "charge", "interest_variable": "capacity"},
    ]
    df = find_parameters_for_section(groups, targets, raw_data=None)
    # Expect 2 columns
    assert df.shape[1] == 2
    # 2 rows total (one row each column), or 2 rows if the code stacks them.
    assert df.shape[0] >= 2
