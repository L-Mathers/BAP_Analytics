# tests/test_feature_extraction.py

import pandas as pd
import pytest
from battery_analysis.feature_extraction import (
    assign_cycle_keys,
    build_ocv_map,
    calculate_coulombic_and_energy_eff,
)


def test_build_ocv_map():
    # Suppose we have one group that is a full discharge
    groups = [
        {
            "group_type": "discharge",
            "full_cycle": True,
            "start_index": 0,
            "end_index": 4,
            "c-rate": 0.05,
        }
    ]

    df = pd.DataFrame(
        {"discharge_capacity": [0, 0.5, 1.0, 1.5, 2.0], "voltage": [4.2, 4.0, 3.8, 3.4, 3.1]}
    )

    ocv_func = build_ocv_map(groups, df, nominal_capacity=2.0)
    # Interpolate a voltage in between
    soc_val = ocv_func(3.7)
    assert soc_val is not None


def test_assign_cycle_keys():
    data = [
        {"group_type": "charge"},
        {"group_type": "discharge", "full_cycle": True},
        {"group_type": "charge"},
        {"group_type": "discharge", "full_cycle": True},
    ]
    data, max_cycle = assign_cycle_keys(data, is_rpt=False)
    # We should have 2 cycles assigned
    assert max_cycle == 2


def test_calculate_coulombic_and_energy_eff():
    data = [
        {
            "group_type": "charge",
            "cycle": 1,
            "capacity": 1.2,
            "energy": 4.2,
        },
        {
            "group_type": "discharge",
            "cycle": 1,
            "capacity": 1.1,
            "energy": 3.9,
        },
    ]
    calculate_coulombic_and_energy_eff(data, is_rpt=False, capacity=1.2)
    # Efficiency should be about 91.66... for coulombic, 92.85... for energy
    charge_grp = data[0]
    discharge_grp = data[1]
    assert 90 < charge_grp["coulombic_efficiency"] < 95
    assert 90 < charge_grp["energy_efficiency"] < 95
