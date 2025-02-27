# processing.py

import pandas as pd
import numpy as np

# Fuzzy matching (if not already in a dedicated data_processing function)
from fuzzywuzzy import process, fuzz

# Import from your refactored modules
from processing_library.config_builder import build_config_for_test_type
from processing_library.data_processing import (
    psuedo_limit_extraction,
    add_cv_steps
)
from processing_library.analysis_aggregator import find_parameters_for_section
from processing_library.feature_extraction import (
    build_ocv_map,
    assign_cycle_keys,
    calculate_coulombic_and_energy_eff,
    estimate_soc

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
    print(df.columns)
    # Add SOC column
    df = estimate_soc(df)
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
        print(f"Group {gnum}: Type={group_type}, Start Voltage={start_voltage}, End Voltage={end_voltage}, Start SOC={start_soc}, End SOC={end_soc}")
    # Throughput trackers
    ch_en_thru = 0.0
    dch_en_thru = 0.0
    ch_cap_thru = 0.0
    dch_cap_thru = 0.0
    first_dch_flag = False

    # Some voltage thresholds used to determine "full cycle"
    high_thr = pseudo_high * 0.95
    low_thr = pseudo_low * 1.05

    group_data = []

    # 3) Build group_data for each group
    for gnum, gdf in grouped:
        start_i = gdf["original_index"].iloc[0]
        end_i = gdf["original_index"].iloc[-1]
        dur = round(gdf["time"].iloc[-1] - gdf["time"].iloc[0], -1)

        init_ch_cap = gdf["charge_capacity"].iloc[0]
        init_dch_cap = gdf["discharge_capacity"].iloc[0]
        init_ch_en = gdf["charge_energy"].iloc[0]
        init_dch_en = gdf["discharge_energy"].iloc[0]
        group_dict = {
            "group_number": gnum,
            "start_index": start_i,
            "end_index": end_i,
            "duration": dur,
            "start_voltage": gdf["voltage"].iloc[0],
            "end_voltage": gdf["voltage"].iloc[-1],
            "capacity": None,
            "energy": None,
            "pulse": (dur < 40),  # Arbitrary definition
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
            "c-rate": None,
            "soc": gdf['soc'].iloc[0],
            "group_type": None,
            "ch_energy_throughput": None,
            "dch_energy_throughput": None,
            "total_energy_throughput": None,
            "ch_capacity_throughput": None,
            "dch_capacity_throughput": None,
            "total_capacity_throughput": None,
            "CCCV": False,
        }

        # Detect CC-CV flag
        unique_steps = gdf["Step Type"].unique()
        cccv_flag = any(s in unique_steps for s in ["charge cv", "discharge cv"])
        group_dict["CCCV"] = cccv_flag

        # Evaluate c-rate based on main portion
        if capacity and capacity > 0:
            if cccv_flag:
                cc_df = gdf[gdf["Step Type"].isin(["charge", "discharge"])]
                group_dict["c-rate"] = round(abs(cc_df["current"].mean()) / capacity, 3)
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
                        cv_cap = abs(gdf.loc[cv_end, "charge_capacity"] - gdf.loc[cv_start, "charge_capacity"])
                        cv_en = abs(gdf.loc[cv_end, "charge_energy"] - gdf.loc[cv_start, "charge_energy"])
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
                        cv_cap = abs(gdf.loc[cv_end, "discharge_capacity"] - gdf.loc[cv_start, "discharge_capacity"])
                        cv_en = abs(gdf.loc[cv_end, "discharge_energy"] - gdf.loc[cv_start, "discharge_energy"])
                        group_dict["cc_capacity"] = cc_cap
                        group_dict["cv_capacity"] = cv_cap
                        group_dict["cc_energy"] = cc_en
                        group_dict["cv_energy"] = cv_en

            else:
                group_dict["c-rate"] = round(abs(gdf["current"].mean()) / capacity, 3)
                group_dict["ave_cc_u"] = gdf["voltage"].mean()
        else:
            group_dict["c-rate"] = 0.0
            group_dict["ave_cc_u"] = gdf["voltage"].mean()

        # Identify phase
        is_charge = gdf["Step Type"].str.contains(r"\bcharge\b", case=False, regex=True).any()
        is_discharge = gdf["Step Type"].str.contains(r"\bdischarge\b", case=False, regex=True).any()

        if is_charge:
            group_dict["group_type"] = "charge"
            full_c = (group_dict["start_voltage"] <= low_thr) and (group_dict["end_voltage"] >= high_thr)
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

        elif is_discharge:
            group_dict["group_type"] = "discharge"
            full_c = (group_dict["start_voltage"] >= high_thr) and (group_dict["end_voltage"] <= low_thr)
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

        else:
            group_dict["group_type"] = "rest"
            # Possibly mark as calendar if dur > 15 days
            if dur > 1296000:
                group_dict["group_type"] = "calendar"
                group_dict["voltage_drop"] = group_dict["start_voltage"] - group_dict["end_voltage"]

        group_data.append(group_dict)

    # 4) Build OCV map
    pulse_durations = user_input.get("pulse_durations", [10, 2, 0.1]) if user_input else [10]

    # ### SECOND PASS FOR PULSE CALCULATIONS ###
    for gd in group_data:
        if gd["pulse"]:
            s_i, e_i = gd["start_index"], gd["end_index"]
            sub_df = df.loc[s_i:e_i]
            v1 = sub_df["voltage"].iloc[0]
            I = sub_df["current"].mean()
            for dur in pulse_durations:
                start_t = sub_df["time"].iloc[0]
                cutoff = start_t + dur
                after = sub_df[sub_df["time"] >= cutoff]
                if not after.empty:
                    v2 = after["voltage"].iloc[0]
                    dv = v1 - v2
                    if abs(I) > 1e-6:
                        ohms = dv / abs(I)
                        mOhms = ohms * 1000
                        gd[f"internal_resistance_{dur}s"] = mOhms
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

    # 5) Assign cycles
    group_data, max_cyc = assign_cycle_keys(group_data, is_rpt)

    # Summaries
    total_dur = 0
    if group_data:
        last_end = group_data[-1]["end_index"]
        total_dur = df.loc[last_end, "time"] / 86400.0

    total_en_thru = (ch_en_thru + dch_en_thru) / 1000
    total_cap_thru = (ch_cap_thru + dch_cap_thru) / 1000
    eq_cycle = (dch_cap_thru / capacity) if capacity else None

    summary = {
        "group_number": len(group_data) + 2,
        "group_type": "summary",
        "max_cycles": max_cyc,
        "total_duration": total_dur,
        "ch_energy_throughput": ch_en_thru / 1000,
        "dch_energy_throughput": dch_en_thru / 1000,
        "total_energy_throughput": total_en_thru,
        "ch_capacity_throughput": ch_cap_thru / 1000,
        "dch_capacity_throughput": dch_cap_thru / 1000,
        "total_capacity_throughput": total_cap_thru,
        "eq_cycle": eq_cycle,
    }
    group_data.append(summary)
   
    
    group_data = calculate_coulombic_and_energy_eff(group_data)

    # 6) Build final DataFrame from config targets
    # If test_type not in config targets, default to empty list
    targets_for_test = config.get("targets", {}).get(test_type, [])

    print('calling find_parameters_for_section')
    
    final_df = find_parameters_for_section(group_data, targets_for_test, raw_data=df)

    return final_df


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
    is_rpt = (test_type == "Rate Performance Test")

    # 3) Fuzzy-match required columns
    required_cols = {
        "time": ["time","elapsed time","test time","time (s)"],
        "current": ["current","current (a)", 'I[A]'],
        "voltage": ["voltage","voltage (v)", 'U[V]'],
        "capacity": ["capacity","capacity (ah)","capacity (mah)"],
        "discharge_capacity": ["discharge capacity (ah)","dcap","discharge_capacity", "Ah-Ch-Set" ],
        "charge_capacity": ["charge capacity (ah)","ccap","charge_capacity", 'Ah-Dch-Set'],
        "discharge_energy": ["discharge energy (wh)","denergy","discharge_energy"],
        "charge_energy": ["charge energy (wh)","cenergy","charge_energy"]
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
            print(f"Matched {best_match} to {cname}")
        else:
            raise ValueError(f"Missing or unmatched column for {cname}")

    # 4) Rename columns
    data = data.rename(columns=matched_columns)

    # 5) Pick temperature column if present
    temp_cols = [c for c in data.columns if "temp" in c.lower() or "aux" in c.lower() or 't1' in c.lower()]
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
        user_input=user_input
    )

    

    # 7) Remove columns that are entirely NaN
    result_df = result_df.dropna(axis=1, how="all")

    # Optionally save the final KPI results to CSV
    result_df.to_csv("inclusive_kpi.csv", index=False)

    return result_df