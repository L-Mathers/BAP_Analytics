# config_builder.py

import copy


def build_dynamic_rpt_config(base_config, user_input):
    """
    Extends the RPT (Rate Performance) config with
    user-provided pulses, special crates, etc.
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
        rpt_targets.append(
            {
                "key": f"DCIR_{dur}s",
                "group_type": "discharge",
                "pulse": True,
                "interest_variable": f"internal_resistance_{dur}s",
                "per_cycle": False,
                "test_type": "rpt",
            }
        )
        rpt_targets.append(
            {
                "key": f"SOC",
                "group_type": "discharge",
                "pulse": True,
                "interest_variable": "soc",
                "per_cycle": False,
                "test_type": "rpt",
            }
        )

        rpt_targets.append(
            {
                "key": "cycle",
                "group_type": "discharge",
                "interest_variable": "cycle",
                "pulse": True,
                "per_cycle": False,
                "test_type": "rpt",
            }
        )

    for crate in user_input.get("special_crates", []):
        rpt_targets.append(
            {
                "key": f"{crate}C_Dch_Tmax",
                "crate": [crate],
                "group_type": "discharge",
                "pulse": False,
                "interest_variable": "max_temp",
                "per_cycle": False,
                "test_type": "rpt",
            }
        )
        rpt_targets.append(
            {
                "key": f"{crate}C_Dch_Capacity",
                "crate": [crate],
                "group_type": "discharge",
                "pulse": False,
                "interest_variable": "capacity",
                "per_cycle": False,
                "test_type": "rpt",
            }
        )
        rpt_targets.append(
            {
                "key": f"{crate}C_Dch_duration",
                "crate": [crate],
                "group_type": "discharge",
                "pulse": False,
                "interest_variable": "duration",
                "per_cycle": False,
                "test_type": "rpt",
            }
        )
        rpt_targets.append(
            {
                "key": "cycle",
                "group_type": "discharge",
                "crate": [crate],
                "interest_variable": "cycle",
                "pulse": False,
                "per_cycle": False,
                "test_type": "rpt",
            }
        )

    # Get dcir_normalization parameter
    dcir_norm = user_input.get("dcir_normalization", None)

    # Only add normalized resistance target if dcir_norm is non-empty and has 2 values
    if dcir_norm and len(dcir_norm) == 2:
        soc, dur = dcir_norm
        rpt_targets.append(
            {
                "key": f"normalized_internal_resistance_{dur}s",
                "group_type": "discharge",
                "pulse": True,
                "interest_variable": f"normalized_internal_resistance_{dur}s",
                "per_cycle": False,
                "test_type": "rpt",
            }
        )
    if user_input.get("pocv", False):
        rpt_targets.append(
            {
                "key": "pocv_voltage",
                "group_type": "discharge",
                "interest_variable": "voltage",
                "crate": [0.01, 0.1],
                "per_cycle": False,
                "time_series": True,
                "test_type": "rpt",
            }
        ),
        rpt_targets.append(
            {
                "key": "pocv_soc",
                "group_type": "discharge",
                "interest_variable": "soc",
                "crate": [0.01, 0.1],
                "per_cycle": False,
                "time_series": True,
                "test_type": "rpt",
            }
        ),
        rpt_targets.append(
            {
                "key": "pocv_time",
                "group_type": "discharge",
                "interest_variable": "time",
                "crate": [0.01, 0.1],
                "per_cycle": False,
                "time_series": True,
                "test_type": "rpt",
            }
        )

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

    aging_targets = updated_config["targets"]["Cycle Aging"]

    # check for normalization by nominal capacity
    if user_input.get("nominal_normalization", False):
        aging_targets.append(
            {
                "key": "nominal_normalized_capacity",
                "group_type": "discharge",
                "interest_variable": "nominal_normalized_capacity",
                "per_cycle": True,
                "test_type": "cycling",
            }
        )

    # check for normalization by first cycle capacity
    if user_input.get("first_cycle_normalization", False):
        aging_targets.append(
            {
                "key": "first_cycle_normalized_capacity",
                "group_type": "discharge",
                "interest_variable": "first_cycle_normalized_capacity",
                "per_cycle": True,
                "test_type": "cycling",
            }
        )

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
        cal_targets.append(
            {
                "key": custom_key,
                "group_type": "rest",
                "time_series": True,
                "interest_variable": "voltage",
            }
        )

    return updated_config


def build_config_for_test_type(base_config, test_type, user_input):
    """
    Given a test type and user input, build the final config.
    """
    if test_type == "Rate Performance Test":
        final_config = build_dynamic_rpt_config(base_config, user_input)
    elif test_type == "Cycle Aging":
        final_config = build_dynamic_aging_config(base_config, user_input)
    elif test_type == "Calendar Aging":
        final_config = build_dynamic_calendar_config(base_config, user_input)
    elif test_type == "Combined RPT/Cycling":
        rpt_config = build_dynamic_rpt_config(base_config, user_input)
        aging_config = build_dynamic_aging_config(base_config, user_input)
        final_config = {
            "targets": {
                "Combined RPT/Cycling": rpt_config["targets"]["Rate Performance Test"]
                + aging_config["targets"]["Cycle Aging"]
            }
        }

    else:
        # No dynamic changes - just use the base config
        final_config = base_config

    return final_config
