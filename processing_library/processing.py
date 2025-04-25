# processing.py

import numpy as np
import pandas as pd

# Fuzzy matching (if not already in a dedicated data_processing function)
from fuzzywuzzy import fuzz, process

from processing_library.analysis_aggregator import find_parameters_for_section

# Import from your refactored modules
from processing_library.config_builder import build_config_for_test_type
from processing_library.custom_processing import dcir_processing
from processing_library.feature_extraction import (
    add_cv_steps,
    assign_cycle_keys,
    calculate_coulombic_and_energy_eff,
    create_merged_capacity,
    estimate_soc,
    normalize_capacity,
    normalize_dcir,
    psuedo_limit_extraction,
    seperate_test_types,
)


def data_extractor(df, capacity, config, test_type, is_rpt, user_input=None):
    """
    Main grouping & labeling, then calls 'find_parameters_for_section'
    to extract the newly defined metrics.

    :param df: raw DataFrame with (time, current, voltage, capacity, energy, etc.)
    :param capacity: nominal capacity
    :param config: the final config containing only the new targets
    :param test_type: e.g. "Rate Performance Test"
    :param is_rpt: bool
    :param user_input: dictionary with "pulse_durations", "voltage_relaxation", etc.
    """
    # 1) Identify charge/discharge/rest + pseudo-limits
    df, pseudo_high, pseudo_low = psuedo_limit_extraction(df)
    # 2) Add CV steps
    df = add_cv_steps(df, vhigh=pseudo_high, vlow=pseudo_low)
    # Add SOC column

    df = estimate_soc(df, nom_cap=capacity)
    # Keep track of original index
    df["original_index"] = df.index
    # Group the data
    grouped = df.groupby("group")
    for gnum, gdf in grouped:
        start_voltage = gdf["voltage"].iloc[0]
        end_voltage = gdf["voltage"].iloc[-1]
        start_soc = gdf["soc"].iloc[0]
        end_soc = gdf["soc"].iloc[-1]
        group_type = gdf["Step Type"].iloc[0]
    # Throughput trackers
    ch_en_thru = 0.0
    dch_en_thru = 0.0
    ch_cap_thru = 0.0
    dch_cap_thru = 0.0
    first_dch_flag = False
    first_pulse = False
    first_dcir = 0.0

    # Some voltage thresholds used to determine "full cycle"
    high_thr = pseudo_high * 0.95
    low_thr = pseudo_low * 1.05

    dcir_normalization = user_input.get("dcir_normalization", None)
    if dcir_normalization and len(dcir_normalization) != 2:
        print(
            f"Warning: Invalid dcir_normalization format: {dcir_normalization}. Should be [SOC, duration]"
        )
        dcir_normalization = None

    nominal_normalization = user_input.get("nominal_normalization", False)
    first_cycle_normalization = user_input.get("first_cycle_normalization", False)
    group_data = []

    # Track previous charge and discharge groups for full cycle detection
    prev_charge_end_voltage = None
    prev_discharge_end_voltage = None

    # 3) Build group_data for each group
    for gnum, gdf in grouped:
        start_i = gdf["original_index"].iloc[0]
        end_i = gdf["original_index"].iloc[-1]
        dur = round(gdf["time"].iloc[-1] - gdf["time"].iloc[0], -1)

        init_ch_cap = gdf["charge_capacity"].iloc[0]
        init_dch_cap = gdf["discharge_capacity"].iloc[0]
        init_ch_en = gdf["charge_energy"].iloc[0]
        init_dch_en = gdf["discharge_energy"].iloc[0]

        start_voltage = gdf["voltage"].iloc[0]
        end_voltage = gdf["voltage"].iloc[-1]

        group_dict = {
            "group_number": gnum,
            "start_index": start_i,
            "end_index": end_i,
            "duration": dur,
            "start_voltage": start_voltage,
            "end_voltage": end_voltage,
            "capacity": None,
            "energy": None,
            "pulse": (dur < 60 and dur > 5),
            "full_cycle": False,
            "cc_capacity": None,
            "cv_capacity": None,
            "cc_energy": None,
            "cv_energy": None,
            "max_temp": gdf["temperature"].max() if "temperature" in gdf.columns else None,
            "u_min": gdf["voltage"].min(),
            "u_max": gdf["voltage"].max(),
            "i_max": gdf["current"].max(),
            "i_min": gdf["current"].min(),
            "ave_cc_u": None,
            "relative_capacity": None,
            "relative_energy": None,
            "coulombic_efficiency": None,
            "energy_efficiency": None,
            "crate": gdf["current"].mean() / capacity if capacity else None,
            "soc": round(gdf["soc"].iloc[0]),
            "group_type": None,
            "ch_energy_throughput": None,
            "dch_energy_throughput": None,
            "total_energy_throughput": None,
            "ch_capacity_throughput": None,
            "dch_capacity_throughput": None,
            "total_capacity_throughput": None,
            "CCCV": False,
            "cycle": 0,
            "nominal_normalized_capacity": None,
            "first_cycle_normalized_capacity": None,
        }

        # Detect CC-CV flag
        unique_steps = gdf["Step Type"].unique()
        cccv_flag = any(s in unique_steps for s in ["charge cv", "discharge cv"])
        group_dict["CCCV"] = cccv_flag

        # Evaluate crate based on main portion
        if capacity and capacity > 0:
            if cccv_flag:
                cc_df = gdf[gdf["Step Type"].isin(["charge", "discharge"])]
                group_dict["crate"] = round(abs(cc_df["current"].mean()) / capacity, 2)
                cv_idxs = gdf[gdf["Step Type"].isin(["charge cv", "discharge cv"])].index
                if len(cv_idxs) > 0:
                    cv_start = cv_idxs[0]
                    cv_end = cv_idxs[-1]
                    group_dict["cv_start"] = cv_start
                    group_dict["cv_end"] = cv_end
                    group_dict["ave_cc_u"] = gdf.loc[start_i:cv_end, "voltage"].mean()

                    # Distinguish charge vs discharge CV
                    if "charge cv" in unique_steps:
                        group_dict["capacity"] = abs(gdf["charge_capacity"].max() - init_ch_cap)
                        group_dict["energy"] = abs(gdf["charge_energy"].max() - init_ch_en)

                        cc_cap = abs(gdf.loc[cv_start, "charge_capacity"] - init_ch_cap)
                        cc_en = abs(gdf.loc[cv_start, "charge_energy"] - init_ch_en)
                        cv_cap = abs(
                            gdf.loc[cv_end, "charge_capacity"]
                            - gdf.loc[cv_start, "charge_capacity"]
                        )
                        cv_en = abs(
                            gdf.loc[cv_end, "charge_energy"] - gdf.loc[cv_start, "charge_energy"]
                        )
                        group_dict["cc_capacity"] = cc_cap
                        group_dict["cv_capacity"] = cv_cap
                        group_dict["cc_energy"] = cc_en
                        group_dict["cv_energy"] = cv_en
                    else:
                        # discharge CV
                        group_dict["capacity"] = abs(gdf["discharge_capacity"].max() - init_dch_cap)
                        group_dict["energy"] = abs(gdf["discharge_energy"].max() - init_dch_en)

                        cc_cap = abs(gdf.loc[cv_start, "discharge_capacity"] - init_dch_cap)
                        cc_en = abs(gdf.loc[cv_start, "discharge_energy"] - init_dch_en)
                        cv_cap = abs(
                            gdf.loc[cv_end, "discharge_capacity"]
                            - gdf.loc[cv_start, "discharge_capacity"]
                        )
                        cv_en = abs(
                            gdf.loc[cv_end, "discharge_energy"]
                            - gdf.loc[cv_start, "discharge_energy"]
                        )
                        group_dict["cc_capacity"] = cc_cap
                        group_dict["cv_capacity"] = cv_cap
                        group_dict["cc_energy"] = cc_en
                        group_dict["cv_energy"] = cv_en

            else:
                group_dict["crate"] = round(abs(gdf["current"].mean()) / capacity, 2)
                group_dict["ave_cc_u"] = gdf["voltage"].mean()
        else:
            group_dict["crate"] = 0.0
            group_dict["ave_cc_u"] = gdf["voltage"].mean()

        # Identify phase
        is_charge = gdf["Step Type"].str.contains(r"\bcharge\b", case=False, regex=True).any()
        is_discharge = gdf["Step Type"].str.contains(r"\bdischarge\b", case=False, regex=True).any()

        if is_charge:
            group_dict["group_type"] = "charge"
            # Check for full cycle using previous discharge group's start voltage if available
            if prev_discharge_end_voltage is not None:
                full_c = (prev_discharge_end_voltage <= low_thr) and (end_voltage >= high_thr)

            else:
                full_c = (start_voltage <= low_thr) and (end_voltage >= high_thr)
            group_dict["full_cycle"] = full_c

            # If no CV portion found
            if group_dict["cc_capacity"] is None:
                group_dict["capacity"] = abs(gdf["charge_capacity"].max() - init_ch_cap)
                group_dict["energy"] = abs(gdf["charge_energy"].max() - init_ch_en)
                group_dict["cc_capacity"] = group_dict["capacity"]
                group_dict["cc_energy"] = group_dict["energy"]

            ch_en_thru += group_dict["energy"]
            ch_cap_thru += group_dict["capacity"]
            group_dict["ch_energy_throughput"] = ch_en_thru / 1000
            group_dict["ch_capacity_throughput"] = ch_cap_thru / 1000

            # Update previous charge start voltage
            prev_charge_end_voltage = end_voltage

        elif is_discharge:
            group_dict["group_type"] = "discharge"
            # Check for full cycle using previous charge group's start voltage if available
            if prev_charge_end_voltage is not None:
                full_c = (prev_charge_end_voltage >= high_thr) and (end_voltage <= low_thr)
            else:
                full_c = (start_voltage >= high_thr) and (end_voltage <= low_thr)
            group_dict["full_cycle"] = full_c

            if group_dict["cc_capacity"] is None:
                group_dict["capacity"] = abs(gdf["discharge_capacity"].max() - init_dch_cap)
                group_dict["energy"] = abs(gdf["discharge_energy"].max() - init_dch_en)
                group_dict["cc_capacity"] = group_dict["capacity"]
                group_dict["cc_energy"] = group_dict["energy"]

            dch_en_thru += group_dict["energy"]
            dch_cap_thru += group_dict["capacity"]
            group_dict["dch_energy_throughput"] = dch_en_thru / 1000
            group_dict["dch_capacity_throughput"] = dch_cap_thru / 1000

            # Relative capacity referencing the first discharge
            if not first_dch_flag:
                first_cap = group_dict["capacity"]
                first_en = group_dict["energy"]
                group_dict["relative_capacity"] = 100
                group_dict["relative_energy"] = 100
                first_dch_flag = True
            else:
                group_dict["relative_capacity"] = (group_dict["capacity"] / first_cap) * 100
                group_dict["relative_energy"] = (group_dict["energy"] / first_en) * 100

            # Update previous discharge start voltage
            prev_discharge_end_voltage = end_voltage

        else:
            group_dict["group_type"] = "rest"
            # Possibly mark as calendar if dur > 15 days
            if dur > 1296000:
                group_dict["group_type"] = "calendar"
                group_dict["voltage_drop"] = group_dict["start_voltage"] - group_dict["end_voltage"]

        group_data.append(group_dict)
        # No longer need this since we're tracking charge and discharge separately
        # prev_group_start_voltage = start_voltage

    pulse_durations = user_input.get("pulse_durations", [10, 2, 0.1]) if user_input else [10]
    for gd in group_data:
        if gd["pulse"]:

            s_i, e_i = gd["start_index"], gd["end_index"]
            for dur in pulse_durations:
                dcir = dcir_processing(df, s_i, e_i, dur, client="daimler_truck")
                gd[f"internal_resistance_{dur}s"] = dcir
                # sub_df = df.loc[s_i:e_i]
                # v1 = df["voltage"].iloc[gd["start_index"]-1]
                # pulsev = sub_df["voltage"].iloc[0]
                # print('current group', v1)
                # print('-----------')
                # print('prev group', pulsev)
                # print('-----------')
                # print('sub df', sub_df[['voltage', 'time', 'current']])
                # I = sub_df["current"].max()
                # for dur in pulse_durations:
                #     start_t = sub_df["time"].iloc[0]
                #     cutoff = start_t + dur
                #     after = sub_df[sub_df["time"] >= cutoff]
                #     if not after.empty:
                #         v2 = after["voltage"].iloc[0]
                #         dv = v1 - v2
                #         if abs(I) > 1e-6:
                #             ohms = dv / abs(I)
                #             mOhms = ohms * 1000
                #             gd[f"internal_resistance_{dur}s"] = mOhms
                #             print(f"Internal resistance for {dur}s pulse: {mOhms} mOhm with current {I}A at {v1}V")

                # Check for first pulse for normalization
                if dcir_normalization and len(dcir_normalization) == 2:
                    target_dur = dcir_normalization[1]
                    target_soc = dcir_normalization[0]
                    if dur == target_dur and abs(gd["soc"] - target_soc) <= 1 and not first_pulse:
                        first_dcir = dcir
                        first_pulse = True
            else:
                gd[f"internal_resistance_{dur}s"] = None

    # ### VOLTAGE RELAXATION AFTER PULSE ###
    if user_input and user_input.get("voltage_relaxation", False):
        # Another pass to check if rest group follows a pulse
        for i, gd in enumerate(group_data):
            if gd["pulse"] and (i + 1 < len(group_data)):
                nxt = group_data[i + 1]
                if nxt["group_type"] == "rest":
                    nxt["group_type"] = "relaxation"
                    s_i2, e_i2 = nxt["start_index"], nxt["end_index"]
                    # If needed, store data or indexes
                    gd["relaxation_start"] = s_i2
                    gd["relaxation_end"] = e_i2

    # If dcir_normalization is specified, normalize the values after all groups are processed
    if dcir_normalization and len(dcir_normalization) == 2 and first_pulse and first_dcir > 0:
        group_data = normalize_dcir(group_data, first_dcir, dcir_normalization)

    # Classify groups as cycling or rpt
    group_data = seperate_test_types(group_data, test_type)

    # Assign cycle keys
    group_data, _ = assign_cycle_keys(group_data)
    # Calculate coulombic and energy efficiency
    group_data = calculate_coulombic_and_energy_eff(group_data)

    group_data = normalize_capacity(
        group_data,
        user_input.get("nominal_normalization", False),
        user_input.get("first_cycle_normalization", False),
        capacity,
    )

    # Extract final metrics using the targets from the config
    section_targets = config["targets"].get(test_type, [])
    df_metrics = find_parameters_for_section(
        group_data, section_targets, c_rate_tolerance=0.5, raw_data=df
    )

    return df_metrics


def process_lifetime_test(data: pd.DataFrame, combined_input: dict, config: dict):
    """
    1) Column fuzzy-match
    2) data_extractor
    3) returns final DataFrame results
    """
    # 1) Extract user parameters
    test_type = combined_input.get("test_type", None)
    capacity = combined_input.get("cell_limits", {}).get("capacity", None)
    user_input = combined_input.get("user_input", {})
    # 2) Merge static + dynamic config
    final_config = build_config_for_test_type(config, test_type, user_input)
    is_rpt = test_type == "Rate Performance Test" or test_type == "Combined RPT/Cycling"

    # 3) Fuzzy-match required columns
    required_cols = {
        "time": ["time", "elapsed time", "test time", "time (s)"],
        "current": ["current", "current (a)", "I[A]"],
        "voltage": ["voltage", "voltage (v)", "U[V]"],
        "capacity": ["capacity", "capacity (ah)", "capacity (mah)"],
        "discharge_capacity": [
            "discharge capacity (ah)",
            "dcap",
            "discharge_capacity",
            "Ah-Dis-Set",
        ],
        "charge_capacity": ["charge capacity (ah)", "ccap", "charge_capacity", "Ah-Ch-Set"],
        "discharge_energy": ["discharge energy (wh)", "denergy", "discharge_energy", "Wh-Dis-Set"],
        "charge_energy": ["charge energy (wh)", "cenergy", "charge_energy", "Wh-Ch-Set"],
    }

    matched_columns = {}
    for cname, possible_list in required_cols.items():
        best_match = None
        best_score = 0
        for candidate in possible_list:
            match_result = process.extractOne(candidate, data.columns, scorer=fuzz.token_sort_ratio)
            if match_result:
                col_name, score = match_result
                if score > best_score:
                    best_score = score
                    best_match = col_name
        if best_match and best_score >= 80:
            matched_columns[best_match] = cname
        else:
            if cname == "capacity":
                continue

            else:
                raise ValueError(f"Missing or unmatched column for {cname}")

    # 4 Rename columns
    data = data.rename(columns=matched_columns)
    # check if time is in seconds
    if data["time"].max() < 10000:
        data["time"] = data["time"] * 3600
        print("converted time to seconds")

    # only take data before 1.4 mil seconds
    # data = data[data["time"] < 1400000]

    # implement zero current tolerance
    data["current"] = data["current"].apply(lambda x: 0 if abs(x) < 0.015 else x)

    data = create_merged_capacity(data)

    # 5) Pick temperature column if present
    temp_cols = [
        c for c in data.columns if "temp" in c.lower() or "aux" in c.lower() or "t1" in c.lower()
    ]
    if temp_cols:
        picked = max(temp_cols, key=lambda x: data[x].max())
        data["temperature"] = data[picked]

    # 6) Run the core extraction logic
    result_df = data_extractor(
        df=data,
        capacity=capacity,
        config=final_config,
        test_type=test_type,
        is_rpt=is_rpt,
        user_input=user_input,
    )

    # 7) Remove columns that are entirely NaN
    result_df = result_df.dropna(axis=1, how="all")

    # Optionally save the final KPI results to CSV
    result_df.to_csv("inclusive_kpi.csv", index=False)

    return result_df
