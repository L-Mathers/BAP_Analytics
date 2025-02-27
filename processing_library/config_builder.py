# config_builder.py

import copy

def build_dynamic_rpt_config(base_config, user_input):
    """
    Extends the RPT (Rate Performance) config with
    user-provided pulses, special c-rates, etc.
    """
    # Copy the base config so we don't mutate it in place
    updated_config = copy.deepcopy(base_config)

    if "targets" not in updated_config:
        updated_config["targets"] = {}
    if "Rate Performance Test" not in updated_config["targets"]:
        updated_config["targets"]["Rate Performance Test"] = []

    rpt_targets = updated_config["targets"]["Rate Performance Test"]

    # Example: add DCIR for each user-defined pulse
    for dur in user_input.get("pulse_durations", []):
        rpt_targets.append({
            "key": f"DCIR_{dur}s",
            "group_type": "discharge",
            "pulse": True,
            "interest_variable": f"internal_resistance_{dur}s",
            "per_cycle": False
        })
        rpt_targets.append({
            "key": f"DCIR_SOC_{dur}s",
            "group_type": "discharge",
            "pulse": True,
            "interest_variable": "soc",
            "per_cycle": False
        })

    # Example: add c-rate specials
    for crate in user_input.get("special_crates", []):
        rpt_targets.append({
            "key": f"{crate}C_Dch_Tmax",
            "cycle": None,
            "c_rate": crate,
            "group_type": "discharge",
            "pulse": False,
            "interest_variable": "max_temp",
            "per_cycle": False
        })
        rpt_targets.append({
            "key": f"{crate}C_Dch_Capacity",
            "cycle": None,
            "c_rate": crate,
            "group_type": "discharge",
            "pulse": False,
            "interest_variable": "capacity",
            "per_cycle": False
        })
        rpt_targets.append({
            "key": f"{crate}C_Dch_duration",
            "cycle": None,
            "c_rate": crate,
            "group_type": "discharge",
            "pulse": False,
            "interest_variable": "duration",
            "per_cycle": False
        })
    return updated_config


def build_dynamic_aging_config(base_config, user_input):
    """
    Extends the existing Aging config with user-defined features if needed.
    """
    updated_config = copy.deepcopy(base_config)

    if "targets" not in updated_config:
        updated_config["targets"] = {}
    if "Cycle Aging" not in updated_config["targets"]:
        updated_config["targets"]["Cycle Aging"] = []

    # For demonstration, we leave it empty.
    return updated_config


def build_dynamic_calendar_config(base_config, user_input):
    """
    Possibly no user-defined additions, or maybe new time-series checks, etc.
    """
    updated_config = copy.deepcopy(base_config)

    if "targets" not in updated_config:
        updated_config["targets"] = {}
    if "Calendar" not in updated_config["targets"]:
        updated_config["targets"]["Calendar"] = []

    cal_targets = updated_config["targets"]["Calendar"]

    # Example: user might want to add a custom time-series check for Calendar
    for custom_key in user_input.get("calendar_timeseries", []):
        cal_targets.append({
            "key": custom_key,
            "group_type": "rest",
            "time_series": True,
            "interest_variable": "voltage"
        })

    return updated_config


def build_config_for_test_type(base_config, test_type, user_input):
    """
    Given a test type and user input, build the final config.
    """
    if test_type == "Rate Performance Test":
        final_config = build_dynamic_rpt_config(base_config, user_input)
    elif test_type == "Cycle Aging":
        final_config = build_dynamic_aging_config(base_config, user_input)
    elif test_type == "Calendar":
        final_config = build_dynamic_calendar_config(base_config, user_input)
    else:
        # No dynamic changes - just use the base config
        final_config = base_config
    return final_config