import numpy as np
import pandas as pd
from fuzzywuzzy import process, fuzz

###############################################################################
# 1) BUILD DYNAMIC CONFIG FOR NEW TARGETS
###############################################################################

def build_dynamic_config(user_input):
    """
    Creates a config for 'RPT' that *only* includes the new parameters:
      1) Pulse durations for internal resistance
      2) Voltage relaxation after pulse
      3) Special C-rates for single-value extraction
      4) pOCV timeseries for c-rate <= 0.1

    We do NOT include any of the old/original targets.
    """

    # Minimal empty base
    config = {
        "targets": {
            "Rate Performance Test": []
        }
    }

    rpt_targets = config["targets"]["Rate Performance Test"]

    # 1) Special C-rate single-value parameters (max_temp, capacity, duration)
    special_crates = user_input.get("special_crates", [])
    for c_rate in special_crates:
        rpt_targets.append({
            "key": f"{c_rate}C_Dch_Tmax",
            "cycle": None,
            "c_rate": c_rate,
            "group_type": "discharge",
            "pulse": False,
            "interest_variable": "max_temp",
            "per_cycle": False
        })
        rpt_targets.append({
            "key": f"{c_rate}C_Dch_Capacity",
            "cycle": None,
            "c_rate": c_rate,
            "group_type": "discharge",
            "pulse": False,
            "interest_variable": "capacity",
            "per_cycle": False
        })
        rpt_targets.append({
            "key": f"{c_rate}C_Dch_duration",
            "cycle": None,
            "c_rate": c_rate,
            "group_type": "discharge",
            "pulse": False,
            "interest_variable": "duration",
            "per_cycle": False
        })

    # 2) Voltage relaxation after pulse, as time_series
    #    if the user requests it:
    if user_input.get("voltage_relaxation", False):
        rpt_targets.append({
            "key": "pulse_relaxation_timeseries",
            "group_type": "relaxation",
            "time_series": True,
            "interest_variable": "voltage",
            "time_series": True

        })

    # 3) pOCV timeseries (for c-rate <= 0.1)
    rpt_targets.append({
        "key": "pOCV_timeseries",
        "range_c_rate": [0.03,0.15],      
        "group_type": "discharge",
        "time_series": True,
        "interest_variable": "voltage"
    })
    # 4) standard cycles
    rpt_targets.append({
        "key": "standard_dch_cycle_capacity",
        "group_type": "discharge",
        "pulse": False,
        "full_cycle": True,
        "range_c_rate": [0.25, 0.5],
        "interest_variable": "capacity",
        "per_cycle": False
        })
    
     # 5) DCIR & SOC for each user-defined pulse duration
    pulse_durations = user_input.get("pulse_durations", [])
    for dur in pulse_durations:
        # DCIR for that duration
        rpt_targets.append({
            "key": f"DCIR_{dur}s",
            "group_type": "discharge",
            "pulse": True,
            "interest_variable": f"internal_resistance_{dur}s",
            "per_cycle": False
        })
        # SOC at that pulse (we can also do separate key if you want them separate)
        rpt_targets.append({
            "key": f"DCIR_SOC_{dur}s",
            "group_type": "discharge",
            "pulse": True,
            "interest_variable": "soc",
            "per_cycle": False
        })

    

    return config

###############################################################################
# 2) CORE LOGIC FOR GROUPING / PROCESSING (mostly unchanged from original)
###############################################################################

def psuedo_limit_extraction(df: pd.DataFrame, zero_current_tolerance: float = 0.1):
    """
    Identifies rest/charge/discharge in the DataFrame using a current threshold.
    Removes short groups (<10 points) and returns pseudo-limits.
    """
    if "current" not in df.columns or "voltage" not in df.columns:
        raise ValueError("Error: Required columns 'current' and 'voltage' not found.")

    current_abs = df["current"].abs()
    current_value = df["current"].where(current_abs > zero_current_tolerance, 0)

    df["Step Type"] = "rest"
    df.loc[current_value > 0, "Step Type"] = "charge"
    df.loc[current_value < 0, "Step Type"] = "discharge"

    df["group"] = (df["Step Type"] != df["Step Type"].shift()).cumsum()
    group_lengths = df.groupby("group").size()

    current_phase = df["Step Type"].iloc[0]
    df["Adjusted Step Type"] = current_phase

    for group_number, length in group_lengths.items():
        indices = df[df["group"] == group_number].index
        proposed = df.loc[indices[0], "Step Type"]
        if length >= 10:
            current_phase = proposed
        df.loc[indices, "Adjusted Step Type"] = current_phase

    df["Step Type"] = df["Adjusted Step Type"]
    df.drop(columns=["Adjusted Step Type"], inplace=True)
    df["group"] = (df["Step Type"] != df["Step Type"].shift()).cumsum()

    # Calculate group durations
    group_durations = df.groupby("group").agg(
        start_time=("time", "first"),
        end_time=("time", "last")
    )
    group_durations["duration"] = group_durations["end_time"] - group_durations["start_time"]
    valid = group_durations[group_durations["duration"] > 120].index
    
    # Filter only for averaging purposes, not df output
    filtered = df[df["group"].isin(valid)]
    
    # Approx pseudo-limits from the filtered region
    avg_last_volt_charge = (
        filtered[filtered["Step Type"] == "charge"]
        .groupby("group")["voltage"]
        .last()
        .mean()
    )
    avg_last_volt_discharge = (
        filtered[filtered["Step Type"] == "discharge"]
        .groupby("group")["voltage"]
        .last()
        .mean()
    )
    df.to_csv('/Users/liammathers/Desktop/Github/BAP_Analytics/intermediate.csv', index=False)
    return (
        df,  # Return full DataFrame
        round(avg_last_volt_charge, 4) if pd.notnull(avg_last_volt_charge) else None,
        round(avg_last_volt_discharge, 4) if pd.notnull(avg_last_volt_discharge) else None,
    )


def add_cv_steps(df: pd.DataFrame, vhigh: float, vlow: float,voltage_tolerance: float = 1e-2, current_delta_threshold: float = 0.05, voltage_delta_threshold: float = 0.0001):
    """
    This function modifies the DataFrame by identifying 'constant voltage (CV)' phases 
    during 'charge' and 'discharge' steps. It updates the 'Step Type' column to reflect 
    when the voltage reaches a certain threshold (vhigh for charge, vlow for discharge) 
    and when the current decreases significantly, indicating the beginning of a CV phase.

    Additionally, it removes any sequences of CV steps that are fewer than 10 consecutive steps.

    Args:
        df (pandas.DataFrame): A DataFrame containing 'Step Type', 'voltage', 'current', and 'group' columns.
        vhigh (float): The voltage threshold for starting 'cv charge'.
        vlow (float): The voltage threshold for starting 'cv discharge'.
        voltage_tolerance (float, optional): The tolerance level for the voltage threshold. Defaults to 1e-2.
        current_delta_threshold (float, optional): The threshold for the change in current to detect CV phase. Defaults to 0.05.

    Returns:
        df (pandas.DataFrame): The modified DataFrame with updated 'Step Type' to include 'charge cv' and 'discharge cv',
        and sequences of fewer than 10 consecutive CV steps removed.
    """

    # Ensure the DataFrame contains the necessary columns
    required_columns = ['Step Type', 'voltage', 'current', 'group']
    if not all(col in df.columns for col in required_columns):
        print("Error: Required columns are missing in the DataFrame.")
        return df

    # Calculate the absolute change in current for each row
    df['delta_current'] = df['current'].diff().abs()
    df['delta_voltage'] = df['voltage'].diff().abs()

    # Process each group of data separately based on the 'group' column
    for group_number, group_data in df.groupby('group'):
        step_type = group_data['Step Type'].iloc[0]  # Get the 'Step Type' of the group

        # Exclude the first 10 rows of the group data
        group_data = group_data.iloc[10:]

        if step_type == 'charge':
            # Condition for identifying the start of 'cv charge' phase:
            # The voltage is close to vhigh and the change in current exceeds the threshold
            # cv_condition = (
            #     (np.abs(group_data['voltage'] - vhigh) <= voltage_tolerance) &
            #     (group_data['delta_current'] >= current_delta_threshold)
            # )
            cv_condition = (
                (group_data['delta_current'] >= current_delta_threshold) &
                (group_data['delta_voltage'] <= voltage_delta_threshold)
            )

            # Find the first index where the CV condition is met
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                # Update 'Step Type' to 'charge cv' from the CV start index onwards in the group
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, 'Step Type'] = 'charge cv'

        elif step_type == 'discharge':
            # Condition for identifying the start of 'cv discharge' phase:
            # The voltage is close to vlow and the change in current exceeds the threshold
            # cv_condition = (
            #     (np.abs(group_data['voltage'] - vlow) <= voltage_tolerance) &
            #     (group_data['delta_current'] >= current_delta_threshold)
            # )
            cv_condition = (
                (group_data['delta_current'] >= current_delta_threshold)
            )

            # Find the first index where the CV condition is met
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                # Update 'Step Type' to 'discharge cv' from the CV start index onwards in the group
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, 'Step Type'] = 'discharge cv'

    # Now, remove CV sequences that are less than 10 consecutive steps
    df['cv_flag'] = (df['Step Type'].str.contains('cv')).fillna(False).astype(int)

    # Identify sequences of CV steps
    df['cv_group'] = (df['cv_flag'] != df['cv_flag'].shift()).cumsum()
    
    # Remove CV sequences that have less than 10 consecutive steps
    for cv_group, group_data in df.groupby('cv_group'):
        if group_data['cv_flag'].iloc[0] == 1 and len(group_data) < 10:
            # Set Step Type back to original (charge or discharge) for short CV sequences
            original_step_type = df.loc[group_data.index[0] - 1, 'Step Type'] if group_data.index[0] > 0 else 'charge' if 'charge cv' in group_data['Step Type'].iloc[0] else 'discharge'
            df.loc[group_data.index, 'Step Type'] = original_step_type
    df.to_csv('test.csv')
    # Remove the temporary columns
    df.drop(columns=['delta_current', 'cv_flag', 'cv_group'], inplace=True)

    return df


def assign_cycle_keys(data, is_rpt):
    """
    Assigns cycle numbers if we have a charge->discharge pairing. 
    If is_rpt=True, we only assign if the discharge is full_cycle.
    """
    last_charge = None
    cycle_number = 0
    for entry in data:
        if entry["group_type"] == "charge":
            last_charge = entry
        elif entry["group_type"] == "discharge" and last_charge:
            if not is_rpt or entry.get("full_cycle", False):
                cycle_number += 1
                last_charge["cycle"] = cycle_number
                entry["cycle"] = cycle_number
                last_charge = None
    return data, cycle_number


###############################################################################
# 3) NEW OCV -> SOC LOGIC (without storing sub-DataFrame)
###############################################################################

def build_ocv_map(groups, df, nominal_capacity):
    """
    Finds the group with the *lowest c-rate* among all FULL discharge cycles,
    and uses start_index / end_index to extract data from df for the OCV map.
    Returns a function ocv_to_soc(voltage).

    If none found, returns a dummy that returns None.
    """

    full_dch = [g for g in groups if g.get("group_type") == "discharge" and g.get("full_cycle", False)]
    if not full_dch:
        return lambda v: None

    # pick the group with the lowest absolute c-rate
    selected = min(full_dch, key=lambda x: abs(x.get("c-rate", 9999)))
    print(f"Selected group: {selected}")
    sidx = selected["start_index"]
    eidx = selected["end_index"]

    seg = df.loc[sidx:eidx].copy()
    if "discharge_capacity" not in seg.columns or "voltage" not in seg.columns:
        return lambda v: None

    cmin = seg["discharge_capacity"].min()
    cmax = seg["discharge_capacity"].max()
    print(f"Capacity range: {cmin} - {cmax}")
    if np.isclose(cmin, cmax):
        return lambda v: None

    seg["SOC_frac"] = 1.0 - (seg["discharge_capacity"] - cmin)/(cmax - cmin)
    volt_arr = seg["voltage"].values
    soc_arr  = seg["SOC_frac"].values
    volt_arr = volt_arr[::-1]
    soc_arr = soc_arr[::-1]

    def ocv_to_soc(v):
        
        if v <= volt_arr.min():
            return soc_arr[volt_arr.argmin()]
        if v >= volt_arr.max():
            return soc_arr[volt_arr.argmax()]
  
        return np.interp(v, volt_arr, soc_arr)

    return ocv_to_soc


###############################################################################
# 4) MAIN AGGREGATION (find_parameters_for_section) FOR NEW TARGETS
###############################################################################

import numpy as np
import pandas as pd
import math

def find_parameters_for_section(groups, targets, c_rate_tolerance=0.2, raw_data=None):
    """
    Produces a final DataFrame in three vertical 'blocks':
      1) Single-value rows at the top
      2) Per-cycle rows in the middle
      3) Time-series rows at the bottom

    Each partial DataFrame is "wide":  one column per target key, and multiple
    rows for multiple matches. Rows that don't apply to a particular target
    have NaN in that column.

    Then we do pd.concat(..., axis=0) with ignore_index=True.
    """

    single_value_targets = []
    per_cycle_targets = []
    time_series_targets = []

    key_mapping = {
        'c_rate': 'c-rate',
        'range_c_rate': 'c-rate',
        'group_type': 'group_type',
        'full_cycle': 'full_cycle',
        'pulse': 'pulse',
        'cycle': 'cycle',
        'interest_variable': 'interest_variable'
    }

    # Separate the three categories
    for t in targets:
        if t.get('per_cycle', False):
            per_cycle_targets.append(t)
        elif t.get('time_series', False):
            time_series_targets.append(t)
        else:
            single_value_targets.append(t)

    # 1) SINGLE-VALUE PARTIAL DF
    df_sv = _build_single_value_df(groups, single_value_targets, key_mapping, c_rate_tolerance)

    # 2) PER-CYCLE PARTIAL DF
    df_pc = _build_per_cycle_df(groups, per_cycle_targets, key_mapping, c_rate_tolerance)

    # 3) TIME-SERIES PARTIAL DF
    df_ts = _build_time_series_df(groups, time_series_targets, key_mapping, c_rate_tolerance, raw_data)

    # Now vertically concatenate them in the order: single-value, per-cycle, time-series
    # Because each partial DF has columns = union of all target keys, missing columns => NaN
    # We'll unify columns by specifying `join='outer'`.
    # final_df = pd.concat([df_sv, df_pc, df_ts], axis=0, join='outer', ignore_index=True)
    # 2) Collect all partials in a list for convenience
    partial_dfs = [df_sv, df_pc, df_ts]

    # 3) Create the final "column-wise appended" DataFrame
    final_df = columnwise_stack_no_top_padding(partial_dfs)
    if final_df.empty:
        return None
    return final_df
    
    

import numpy as np
import pandas as pd

def columnwise_stack_no_top_padding(dataframes):
    """
    Given a list of DataFrames that share some set of columns (possibly partial),
    build a new wide DataFrame such that:
      - Each column's real data starts at row=0 (no top NaNs).
      - We append the column's data from each DF in the list, top to bottom.
      - Finally, we pad the shorter columns with NaN at the bottom so all columns 
        have the same length in the final DataFrame.
    """

    # 1) Gather union of all columns that appear in any partial DF
    all_cols = set()
    for df in dataframes:
        all_cols.update(df.columns)
    all_cols = sorted(all_cols)  # optional: for stable, alphabetical order

    # 2) For each column, collect data top-to-bottom from each partial.
    col_data = {}
    for col in all_cols:
        collected_vals = []
        for df in dataframes:
            if col in df.columns:
                # Extract the series
                series = df[col]

                # Optionally drop NaN if you don't want to keep partial-DF "placeholders."
                #   Usually we do want to keep real data, but any leading/trailing NaNs
                #   from partial DFs can be removed if you want them truly "packed" at top:
                # series = series.dropna()

                # Extend our collected list
                collected_vals.extend(series.tolist())

        col_data[col] = collected_vals

    # 3) Find the max length among columns
    max_len = max(len(vals) for vals in col_data.values()) if col_data else 0

    # 4) Pad each column’s list with NaN at the bottom
    for col, vals in col_data.items():
        short_by = max_len - len(vals)
        if short_by > 0:
            vals.extend([np.nan]*short_by)

    # 5) Build final wide DF
    final_df = pd.DataFrame(col_data)
    return final_df
###############################################################################
# HELPER to build SINGLE-VALUE partial DF in "wide" format
###############################################################################
def _build_single_value_df(groups, targets, key_mapping, c_rate_tol):
    """
    Builds a wide DataFrame where each single-value target key is its own column,
    and all real data starts at row=0 (no top padding).
    We gather *all matching groups* for each target, top to bottom,
    then pad each column at the *bottom* so all have the same length.

    NOTE: Rows won't align across columns. Row i for Column A is not
          the same group as Row i for Column B.
    """

    import numpy as np
    import pandas as pd

    # 1) Collect all single-value target keys (our final columns).
    all_keys = [t["key"] for t in targets]
    if not all_keys:
        return pd.DataFrame()  # no single-value targets => empty

    # 2) Prepare a list to hold data for each column:
    #    { "target_key": [val, val, val, ...], ... }
    col_data = {k: [] for k in all_keys}

    # 3) For each target (i.e. each column), gather matches from groups
    for tgt in targets:
        tkey = tgt["key"]
        ivar = tgt.get("interest_variable")
        ignore_keys = {'key','interest_variable','per_cycle','aggregation','time_series'}
        crit_keys  = set(tgt.keys()) - ignore_keys

        # We'll accumulate all values that match this target
        matches_for_this_target = []

        # -- Check every group to see if it matches
        for grp in groups:
            match = True
            for c in crit_keys:
                tval = tgt[c]
                if tval is None:
                    continue  # no constraint on this key
                gkey = key_mapping.get(c, c)
                gval = grp.get(gkey)

                if c == 'c_rate':
                    # within tolerance?
                    if gval is None or abs(gval - tval) > c_rate_tol:
                        match = False
                        break
                elif c == 'range_c_rate':
                    lo, hi = tval
                    if gval is None or not (lo <= gval <= hi):
                        match = False
                        break
                else:
                    # direct equality check
                    if gval != tval:
                        match = False
                        break

            # If matched, collect the group's interest_variable value
            if match:
                val = grp.get(ivar, np.nan)
                matches_for_this_target.append(val)

        # 4) Put this column's matched values into col_data[tkey]
        col_data[tkey].extend(matches_for_this_target)

    # 5) Some columns may have more total matches than others, so we pad.
    max_len = max(len(vals) for vals in col_data.values()) if col_data else 0
    for k, vals in col_data.items():
        diff = max_len - len(vals)
        if diff > 0:
            vals.extend([np.nan]*diff)

    # 6) Construct final DataFrame with columns in all_keys order
    df_sv = pd.DataFrame(col_data, columns=all_keys)
    print(df_sv)
    return df_sv

###############################################################################
# HELPER to build PER-CYCLE partial DF in "wide" format
###############################################################################
def _build_per_cycle_df(groups, targets, key_mapping, c_rate_tol):
    all_keys = [t["key"] for t in targets]
    if not all_keys:
        return pd.DataFrame()

    column_data = {k: [] for k in all_keys}

    # For per-cycle: we produce one row per (cycle, target).
    cycles = sorted(set(g.get("cycle") for g in groups if g.get("cycle") is not None))

    for cyc in cycles:
        cyc_groups = [g for g in groups if g.get("cycle") == cyc]
        for tgt in targets:
            tkey = tgt["key"]
            ivar = tgt.get("interest_variable")
            agg  = tgt.get("aggregation", None)

            ignore_keys = {'key','interest_variable','per_cycle','aggregation','time_series'}
            crit_keys = set(tgt.keys()) - ignore_keys

            matches = []
            for grp in cyc_groups:
                match=True
                for c in crit_keys:
                    tval = tgt[c]
                    gkey = key_mapping.get(c,c)
                    gval = grp.get(gkey)
                    if c=='c_rate':
                        if gval is None or abs(gval - tval) > c_rate_tol:
                            match=False
                            break
                    elif c=='range_c_rate':
                        lo, hi = tval
                        if gval is None or not(lo <= gval <= hi):
                            match=False
                            break
                    else:
                        if gval != tval:
                            match=False
                            break
                if match and ivar in grp:
                    matches.append(grp[ivar])

            if matches:
                if agg=='min':
                    val = min(matches)
                elif agg=='max':
                    val = max(matches)
                elif agg=='sum':
                    val = sum(matches)
                elif agg=='average':
                    val = sum(matches)/ len(matches)
                else:
                    val = matches[0]
            else:
                val = np.nan

            # build row with val in tkey col, NaN in others
            row_dict = {col: np.nan for col in all_keys}
            row_dict[tkey] = val
            for colk in all_keys:
                column_data[colk].append(row_dict[colk])

    # ensure equal length
    max_len = max(len(lst) for lst in column_data.values()) if column_data else 0
    for colk,vlist in column_data.items():
        while len(vlist)< max_len:
            vlist.append(np.nan)

    df_pc = pd.DataFrame(column_data)
    return df_pc


###############################################################################
# HELPER to build TIME-SERIES partial DF in "wide" format
###############################################################################
def _build_time_series_df(groups, targets, key_mapping, c_rate_tol, raw_data):
    all_keys = [t["key"] for t in targets]
    if not all_keys or raw_data is None:
        return pd.DataFrame()

    column_data = {k: [] for k in all_keys}

    for tgt in targets:
        tkey= tgt["key"]
        ivar= tgt.get("interest_variable")

        ignore_keys= {'key','interest_variable','per_cycle','aggregation','time_series'}
        crit_keys= set(tgt.keys()) - ignore_keys

        matched_group= None
        for grp in groups:
            match=True
            for c in crit_keys:
                tval = tgt[c]
                gkey= key_mapping.get(c,c)
                gval= grp.get(gkey)
                if tval is None:
                    continue
                if c=='c_rate' and tval==0.1:
                    if gval is None or gval>(tval + c_rate_tol):
                        match=False
                        break
                elif c=='range_c_rate':
                    lo, hi= tval
                    if gval is None or not(lo<= gval<= hi):
                        match=False
                        break
                else:
                    if gval!= tval:
                        match=False
                        break
            if match:
                matched_group= grp
                break

        if matched_group:
            s_i= matched_group["start_index"]
            e_i= matched_group["end_index"]
            if s_i is not None and e_i is not None and s_i<= e_i:
                sub_df= raw_data.loc[s_i:e_i]
                if ivar and ivar in raw_data.columns:
                    values = sub_df[ivar]
                else:
                    # fallback to "voltage" if present
                    if "voltage" in raw_data.columns:
                        values= sub_df["voltage"]
                    else:
                        values= pd.Series([np.nan]*(e_i - s_i +1), index=sub_df.index)

                for val in values:
                    # new row => val in tkey col
                    row_dict = {col: np.nan for col in all_keys}
                    row_dict[tkey] = val
                    for colk in all_keys:
                        column_data[colk].append(row_dict[colk])
            else:
                # no valid sub-slice => produce 1 row with NaN
                row_dict = {col: np.nan for col in all_keys}
                row_dict[tkey] = np.nan
                for colk in all_keys:
                    column_data[colk].append(row_dict[colk])
        else:
            # no match => 1 row with NaN
            row_dict = {col: np.nan for col in all_keys}
            row_dict[tkey] = np.nan
            for colk in all_keys:
                column_data[colk].append(row_dict[colk])

    # ensure equal length
    max_len = max(len(lst) for lst in column_data.values()) if column_data else 0
    for colk,vlist in column_data.items():
        while len(vlist)< max_len:
            vlist.append(np.nan)

    df_ts = pd.DataFrame(column_data)
    return df_ts


###############################################################################
# 5) DATA_EXTRACTOR (SECOND PASS PULSE LOGIC - no stored sub-DF)
###############################################################################

def data_extractor(df, capacity, config, test_type, is_rpt, user_input=None):
    """
    Main grouping & labeling, then calls 'find_parameters_for_section' 
    to extract the newly defined metrics.

    :param df: raw DataFrame with (time, current, voltage, capacity, energy, etc.)
    :param capacity: nominal capacity
    :param config: the newly built config containing only the new targets
    :param test_type: e.g. "Rate Performance Test"
    :param is_rpt: bool
    :param user_input: dictionary with "pulse_durations", "voltage_relaxation", etc.
    """
    df, pseudo_high, pseudo_low = psuedo_limit_extraction(df)
    

    df = add_cv_steps(df, vhigh=pseudo_high, vlow=pseudo_low)

    # Keep track of original index
    df["original_index"] = df.index
    # df.to_csv('/Users/liammathers/Desktop/Github/BAP_Analytics/processed_data.csv', index=False)
    grouped = df.groupby("group")

    ch_en_thru = 0
    dch_en_thru = 0
    ch_cap_thru = 0
    dch_cap_thru = 0
    first_dch_flag = False

    high_thr = pseudo_high * 0.95
    low_thr  = pseudo_low  * 1.05

    group_data = []
    for gnum, gdf in grouped:
        start_i = gdf["original_index"].iloc[0]
        end_i   = gdf["original_index"].iloc[-1]
        dur     = round(gdf["time"].iloc[-1] - gdf["time"].iloc[0], -1)

        init_ch_cap  = gdf["charge_capacity"].iloc[0]
        init_dch_cap = gdf["discharge_capacity"].iloc[0]
        init_ch_en   = gdf["charge_energy"].iloc[0]
        init_dch_en  = gdf["discharge_energy"].iloc[0]

        group_dict = {
            "group_number": gnum,
            "start_index": start_i,
            "end_index": end_i,
            "duration": dur,
            "start_voltage": gdf["voltage"].iloc[0],
            "end_voltage": gdf["voltage"].iloc[-1],
            "capacity": None,
            "energy": None,
            "pulse": (dur < 40),  # your definition
            "full_cycle": False,
            "cc_capacity": None,
            "cv_capacity": None,
            "cc_energy": None,
            "cv_energy": None,
            "max_temp": gdf["temperature"].max() if "temperature" in gdf.columns else None,
            "u_min": gdf["voltage"].min(),
            "u_max": gdf["voltage"].max(),
            "i_max": gdf["current"].max(),
            "ave_cc_u": None,
            "relative_capacity": None,
            "relative_energy": None,
            "coulombic_efficiency": None,
            "energy_efficiency": None,
            "c-rate": None,
            "group_type": None,
            "ch_energy_throughput": None,
            "dch_energy_throughput": None,
            "total_energy_throughput": None,
            "ch_capacity_throughput": None,
            "dch_capacity_throughput": None,
            "total_capacity_throughput": None,
            "CCCV": False
        }

        unique_steps = gdf["Step Type"].unique()
        cccv_flag = any(s in unique_steps for s in ["charge cv", "discharge cv"])
        group_dict["CCCV"] = cccv_flag

        # Evaluate c-rate from average current of the "main" portion
        if cccv_flag:
            cc_df = gdf[gdf["Step Type"].isin(["charge","discharge"])]
            group_dict["c-rate"] = round(abs(cc_df["current"].mean()) / capacity, 3)
            cv_idxs = gdf[gdf["Step Type"].isin(["charge cv","discharge cv"])].index
            if len(cv_idxs) > 0:
                cv_start = cv_idxs[0]
                cv_end   = cv_idxs[-1]
                group_dict["cv_start"] = cv_start
                group_dict["cv_end"]   = cv_end
                group_dict["ave_cc_u"] = gdf.loc[start_i:cv_end, "voltage"].mean()

                if "charge cv" in unique_steps:
                    group_dict["capacity"] = abs(gdf["charge_capacity"].iloc[-1] - init_ch_cap)
                    group_dict["energy"]   = abs(gdf["charge_energy"].iloc[-1] - init_ch_en)
                    cc_cap = abs(gdf.loc[cv_start,"charge_capacity"] - init_ch_cap)
                    cc_en  = abs(gdf.loc[cv_start,"charge_energy"] - init_ch_en)
                    cv_cap = abs(gdf.loc[cv_end,"charge_capacity"] - gdf.loc[cv_start,"charge_capacity"])
                    cv_en  = abs(gdf.loc[cv_end,"charge_energy"] - gdf.loc[cv_start,"charge_energy"])
                    group_dict["cc_capacity"] = cc_cap
                    group_dict["cv_capacity"] = cv_cap
                    group_dict["cc_energy"]   = cc_en
                    group_dict["cv_energy"]   = cv_en
                else:
                    group_dict["capacity"] = abs(gdf["discharge_capacity"].iloc[-1] - init_dch_cap)
                    group_dict["energy"]   = abs(gdf["discharge_energy"].iloc[-1] - init_dch_en)
                    cc_cap = abs(gdf.loc[cv_start,"discharge_capacity"] - init_dch_cap)
                    cc_en  = abs(gdf.loc[cv_start,"discharge_energy"] - init_dch_en)
                    cv_cap = abs(gdf.loc[cv_end,"discharge_capacity"] - gdf.loc[cv_start,"discharge_capacity"])
                    cv_en  = abs(gdf.loc[cv_end,"discharge_energy"] - gdf.loc[cv_start,"discharge_energy"])
                    group_dict["cc_capacity"] = cc_cap
                    group_dict["cv_capacity"] = cv_cap
                    group_dict["cc_energy"]   = cc_en
                    group_dict["cv_energy"]   = cv_en
        else:
            group_dict["c-rate"] = round(abs(gdf["current"].mean())/capacity, 3)
            group_dict["ave_cc_u"] = gdf["voltage"].mean()

        # Identify phase
        is_charge = gdf["Step Type"].str.contains(r"\bcharge\b", case=False, regex=True).any()
        is_discharge = gdf["Step Type"].str.contains(r"\bdischarge\b", case=False, regex=True).any()

        if is_charge:
            group_dict["group_type"] = "charge"
            full_c = (group_dict["start_voltage"] <= low_thr) and (group_dict["end_voltage"] >= high_thr)
            group_dict["full_cycle"] = full_c
            if group_dict["cc_capacity"] is None:  # no CV
                group_dict["capacity"] = abs(gdf["charge_capacity"].iloc[-1] - init_ch_cap)
                group_dict["energy"]   = abs(gdf["charge_energy"].iloc[-1] - init_ch_en)
                group_dict["cc_capacity"] = group_dict["capacity"]
                group_dict["cc_energy"]   = group_dict["energy"]
            ch_en_thru  += group_dict["energy"]
            ch_cap_thru += group_dict["capacity"]
            group_dict["ch_energy_throughput"]   = ch_en_thru/1000
            group_dict["ch_capacity_throughput"] = ch_cap_thru/1000

        elif is_discharge:
            group_dict["group_type"] = "discharge"
            full_c = (group_dict["start_voltage"] >= high_thr) and (group_dict["end_voltage"] <= low_thr)
            group_dict["full_cycle"] = full_c
            if group_dict["cc_capacity"] is None:
                group_dict["capacity"] = abs(gdf["discharge_capacity"].iloc[-1] - init_dch_cap)
                group_dict["energy"]   = abs(gdf["discharge_energy"].iloc[-1] - init_dch_en)
                group_dict["cc_capacity"] = group_dict["capacity"]
                group_dict["cc_energy"]   = group_dict["energy"]
            dch_en_thru  += group_dict["energy"]
            dch_cap_thru += group_dict["capacity"]
            group_dict["dch_energy_throughput"]   = dch_en_thru/1000
            group_dict["dch_capacity_throughput"] = dch_cap_thru/1000

            if not first_dch_flag:
                first_cap = group_dict["capacity"]
                first_en  = group_dict["energy"]
                group_dict["relative_capacity"] = 100
                group_dict["relative_energy"]   = 100
                first_dch_flag = True
            else:
                group_dict["relative_capacity"] = (group_dict["capacity"]/first_cap)*100
                group_dict["relative_energy"]   = (group_dict["energy"]/first_en)*100

        else:
            group_dict["group_type"] = "rest"

        group_data.append(group_dict)

    # Build OCV map
    ocv_func = build_ocv_map(group_data, df, capacity)
    print(f"OCV function: {ocv_func}")
    pulse_durations = user_input.get("pulse_durations",[10,2,0.1]) if user_input else [10,2,0.1]

    # ### SECOND PASS FOR PULSE CALCULATIONS ###
    for gd in group_data:
        if gd["pulse"]:

            # Let's grab the sub-DataFrame on-the-fly
            s_i, e_i = gd["start_index"], gd["end_index"]
            sub_df = df.loc[s_i:e_i]
            v1  = sub_df["voltage"].iloc[0]
            print(f'first voltage: {v1}')
            soc_approx = ocv_func(v1)
            if soc_approx is not None:
                gd["soc"] = soc_approx * 100
                gd["soc"] = round(soc_approx * 100 / 5) * 5

            I = sub_df["current"].mean()
            for dur in pulse_durations:
                start_t = sub_df["time"].iloc[0]
                cutoff  = start_t + dur
                after   = sub_df[sub_df["time"] >= cutoff]
                if not after.empty:
                    v2 = after["voltage"].iloc[0]
                    dv = v1 - v2
                    # print(f'(v1) {v1} - (v2) {v2}')
                    if abs(I) > 1e-6:
                        ohms  = dv / abs(I)
                        mOhms = ohms * 1000
                        gd[f"internal_resistance_{dur}s"] = mOhms
                     
                    else:
                        gd[f"internal_resistance_{dur}s"] = None

    # ### VOLTAGE RELAXATION AFTER PULSE ###
    if user_input and user_input.get("voltage_relaxation",False):
        # We do another pass to see if a rest group follows
        for i, gd in enumerate(group_data):
            if gd["pulse"]:
                if i+1 < len(group_data):
                    nxt = group_data[i+1]
                    if nxt["group_type"] == "rest":
                        nxt["group_type"] = "relaxation"
                        print(f"Processing relaxation after pulse {gd['group_number']}...")
                        s_i2, e_i2 = nxt["start_index"], nxt["end_index"]
                        rest_df = df.loc[s_i2:e_i2, ["time","voltage"]].copy()
                        print(rest_df)
                        # store as a nested object if you wish,
                        # or just store the indices
                        gd["relaxation_start"] = s_i2
                        gd["relaxation_end"]   = e_i2
                        # If you need the actual data, you can do:
                        # gd["relaxation_df"] = rest_df

    # Assign cycles
    group_data, max_cyc = assign_cycle_keys(group_data, is_rpt)

    # Summaries
    total_dur = 0
    if group_data:
        last_end = group_data[-1]["end_index"]
        total_dur = df.loc[last_end, "time"] / 86400

    total_en_thru = (ch_en_thru + dch_en_thru)/1000
    total_cap_thru= (ch_cap_thru + dch_cap_thru)/1000
    eq_cycle = dch_cap_thru/capacity if capacity else None

    summary = {
        "group_number": len(group_data)+2,
        "group_type": "summary",
        "max_cycles": max_cyc,
        "total_duration": total_dur,
        "ch_energy_throughput": ch_en_thru/1000,
        "dch_energy_throughput": dch_en_thru/1000,
        "total_energy_throughput": total_en_thru,
        "ch_capacity_throughput": ch_cap_thru/1000,
        "dch_capacity_throughput": dch_cap_thru/1000,
        "total_capacity_throughput": total_cap_thru,
        "eq_cycle": eq_cycle
    }
    group_data.append(summary)
    
    # If not RPT, do coulombic / energy eff
    if not is_rpt:
        cyc_map = {}
        for g in group_data:
            c = g.get("cycle")
            if c:
                cyc_map.setdefault(c,[]).append(g)
        for cnum, items in cyc_map.items():
            c_grp = next((x for x in items if x["group_type"]=="charge"),None)
            d_grp = next((x for x in items if x["group_type"]=="discharge"),None)
            if c_grp and d_grp:
                ch_cap = c_grp["capacity"]
                dch_cap= d_grp["capacity"]
                ch_en  = c_grp["energy"]
                dch_en = d_grp["energy"]
                if ch_cap and dch_cap and ch_cap!=0:
                    eff = (dch_cap/ch_cap)*100
                    c_grp["coulombic_efficiency"] = eff
                    d_grp["coulombic_efficiency"] = eff
                if ch_en and dch_en and ch_en!=0:
                    e_eff = (dch_en/ch_en)*100
                    c_grp["energy_efficiency"] = e_eff
                    d_grp["energy_efficiency"] = e_eff

    final_df = find_parameters_for_section(group_data, config["targets"][test_type], raw_data=df)
    return final_df


###############################################################################
# 6) ENTRY POINT: process_lifetime_test
###############################################################################

def process_lifetime_test(data: pd.DataFrame, combined_input: dict):
    """
    1) Column fuzzy-match
    2) data_extractor
    3) returns final results
    """
    test_type = combined_input.get("test_type","Rate Performance Test")
    capacity = combined_input["cell_limits"]["capacity"]
    user_input = combined_input.get("user_input",None)
    config = build_dynamic_config(user_input)
    print(f"CONFIG:{config}")
  
    if test_type == "Rate Performance Test":
        is_rpt = True
        print("Processing Rate Performance Test...")
    else:
        is_rpt = False

    required_cols = {
        "time": ["time","elapsed time","test time","time (s)"],
        "current": ["current","current (a)"],
        "voltage": ["voltage","voltage (v)"],
        "discharge_capacity": ["discharge capacity (ah)","dcap","discharge_capacity"],
        "charge_capacity": ["charge capacity (ah)","ccap","charge_capacity"],
        "discharge_energy": ["discharge energy (wh)","denergy","discharge_energy"],
        "charge_energy": ["charge energy (wh)","cenergy","charge_energy"]
    }
    matched_columns = {}
    for cname, possible in required_cols.items():
        best_match = None
        best_score = 0
        for candidate in possible:
            match = process.extractOne(candidate, data.columns, scorer=fuzz.token_sort_ratio)
            if match and match[1] > best_score:
                best_match = match[0]
                best_score = match[1]
        if best_match and best_score>=80:
            matched_columns[cname] = best_match
            print(f"Matched {cname} to {best_match}")
        else:
            raise ValueError(f"Missing or unmatched column for {cname}")

    data = data.rename(columns={v:k for k,v in matched_columns.items()})

    # find temperature if any
    temp_cols = [c for c in data.columns if "temp" in c.lower() or "aux" in c.lower()]
    if temp_cols:
        # pick the one with max average or something
        picked = max(temp_cols, key=lambda x: data[x].max())
        data["temperature"] = data[picked]
        print(f"Matched temperature to {picked}")

    result_df = data_extractor(
        df=data,
        capacity=capacity,
        config=config,
        test_type=test_type,
        is_rpt=is_rpt,
        user_input=user_input
    )
    return result_df







# # Process
# final_results = process_lifetime_test(df, combined_input, config, user_input)

# print("FINAL RESULTS:")
# print(final_results)

import numpy as np
import pandas as pd
import json
import sys
# sys.path.append('/Users/liammathers/Desktop/Github/BAP_Analytics')
# from RPT_processing import *

from fuzzywuzzy import process, fuzz
import matplotlib.pyplot as plt


# Cycling data
# Iveco
file = "/Users/liammathers/Github/BAP_Analytics/Testing/IV_GO_R115C_005_LP18_549_T25_13-12-2024_AllData.csv" 
# BMW
file = "/Users/liammathers/Github/BAP_Analytics/Testing/BMW_LTF_2580_002_10SOC_Cycling_366_T35_26-06-2024_AllData.csv"
# Electra Vehicles
# file = "/Users/liammathers/Downloads/EV_SKE_556_034_SOC100-0_05C-1C_Deg8_558_T25_27-12-2024_AllData.csv"


#   # Example combined input specifying test type and cell limits
# combined_input = {
#     'test_type': 'Rate Performance Test',
#     'cell_limits': {
#         "capacity": 32.5,
#     },
#     'user_input' : {
#         "pulse_durations": [1, 2, 4],
#         "voltage_relaxation": False,
#         "special_crates": [1.0, 2.0]
#     }
# }

# RPT data
# BMW
file = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/20240726_150811_Z61_EVE_C1_ZPg_D00_SN14524.csv"
# Load data

# cal = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/BMW_LTF_2619_001_Cal1_424_T55_25-06-2024_AllData.csv"
# data = pd.read_csv(cal)


# kpi_rpt = process_lifetime_test(data, combined_input)


