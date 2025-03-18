# data_processing.py

import numpy as np
import pandas as pd
from collections import Counter

def psuedo_limit_extraction(df: pd.DataFrame, zero_current_tolerance: float = 0.1):
    """
    Identifies rest/charge/discharge in the DataFrame using a current threshold.
    Removes short groups (<10 points) and returns pseudo-limits.
    """
    if "current" not in df.columns or "voltage" not in df.columns:
        raise ValueError("Error: Required columns 'current' and 'voltage' not found.")

    current_abs = df["current"].abs()
    current_value = df["current"].where(current_abs > zero_current_tolerance, 0)
    # Set current to zero in the original dataframe where absolute value is below tolerance
    df["current"] = current_value
    # Plot filtered current vs time
   
    vmax = df["voltage"].max()
    vmin = df["voltage"].min()
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
    
    df.to_csv("psuedo.csv")
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

    return (
        df,
        round(avg_last_volt_charge, 4) if pd.notnull(avg_last_volt_charge) else vmax,
        round(avg_last_volt_discharge, 4) if pd.notnull(avg_last_volt_discharge) else vmin,
    )


def add_cv_steps(df: pd.DataFrame,
                 vhigh: float,
                 vlow: float,
                 voltage_tolerance: float = 1e-2,
                 current_delta_threshold: float = 0.05,
                 voltage_delta_threshold: float = 0.0001):
    """
    This function modifies the DataFrame by identifying 'constant voltage (CV)' phases 
    during 'charge' and 'discharge' steps. It updates the 'Step Type' column to reflect 
    CV once a threshold condition is met. Sequences < 10 steps are reverted.
    """
    # Ensure the DataFrame contains the necessary columns
    required_columns = ['Step Type', 'voltage', 'current', 'group']
    if not all(col in df.columns for col in required_columns):
        print("Error: Required columns are missing in the DataFrame.")
        return df

    df['delta_current'] = df['current'].diff().abs()
    df['delta_voltage'] = df['voltage'].diff().abs()

    # Process each group separately
    for group_number, group_data in df.groupby('group'):
        step_type = group_data['Step Type'].iloc[0]
        # Exclude the first 10 rows
        group_data = group_data.iloc[10:]

        if step_type == 'charge':
            # Condition for 'charge cv'
            cv_condition = (
                (group_data['delta_current'] >= current_delta_threshold) &
                (group_data['delta_voltage'] <= voltage_delta_threshold)
            )
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, 'Step Type'] = 'charge cv'

        elif step_type == 'discharge':
            # Condition for 'discharge cv'
            cv_condition = (
                (group_data['delta_current'] >= current_delta_threshold)
            )
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, 'Step Type'] = 'discharge cv'

    # Remove CV sequences that are less than 10 consecutive steps
    df['cv_flag'] = (df['Step Type'].str.contains('cv')).fillna(False).astype(int)
    df['cv_group'] = (df['cv_flag'] != df['cv_flag'].shift()).cumsum()

    for cv_group, group_data in df.groupby('cv_group'):
        if group_data['cv_flag'].iloc[0] == 1 and len(group_data) < 10:
            # Revert to original step type
            # If we want to guess "charge" or "discharge":
            original_type = 'discharge' if 'discharge cv' in group_data['Step Type'].unique()[0] else 'charge'
            df.loc[group_data.index, 'Step Type'] = original_type
    df.to_csv("ccvv.csv")
    df.drop(columns=['delta_current', 'cv_flag', 'cv_group', 'delta_voltage'], inplace=True)
    return df


def match_required_columns(data: pd.DataFrame, threshold: int = 80):
    """
    Fuzzy-match required columns in a DataFrame.
    If a match fails, raises ValueError.
    Returns a dict suitable for `data.rename(columns=...)`.
    """
    from fuzzywuzzy import process, fuzz

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
        if best_match and best_score >= threshold:
            matched_columns[best_match] = cname
        else:
            raise ValueError(f"Missing or unmatched column for {cname}")

    return matched_columns


def create_merged_capacity(
    df: pd.DataFrame,
    charge_col = 'charge_capacity',
    discharge_col = 'discharge_capacity',
    voltage_col = 'voltage',
    combined_col: str = "capacity",
    min_voltage_offset: float = 0.1,
) -> pd.DataFrame:
    """
    1. Classify each row as 'charge', 'discharge', or 'rest' using the provided
       charge_col and discharge_col.
    2. Forward fill zeros in charge/discharge columns within each contiguous step-type group.
    3. Compute combined capacity = abs(charge) - abs(discharge).
    4. Shift the first group's capacity upward if it's an initial discharge that goes negative.
    5. Reset capacity to 0 at the start of a charge group if the previous discharge
       ended within (min_voltage + min_voltage_offset).

    Args:
        df (pd.DataFrame):
            Must contain columns for charge_col, discharge_col, and voltage_col.
        charge_col (str):
            Name of the column containing charge capacity (e.g. 'Q_charge').
        discharge_col (str):
            Name of the column containing discharge capacity (e.g. 'Q_discharge').
        voltage_col (str):
            Name of the column containing voltage.
        combined_col (str, optional):
            Output column name for the combined capacity. Defaults to "capacity".
        min_voltage_offset (float, optional):
            Voltage threshold above the global minimum voltage to reset capacity
            before a new charge. Defaults to 0.05.

    Returns:
        pd.DataFrame:
            A modified copy of the original DataFrame with:
              - 'step_type': str in ['charge', 'discharge', 'rest'] for each row
              - 'group_id': int label identifying contiguous step-type segments
              - combined_col: the final capacity after shifting & reset logic
    """

    df = df.copy()

    # -------------------------------------------------------------
    # 1) STEP TYPE: Identify whether each row is charge, discharge, or rest
    #    using the presence of nonzero charge_col or discharge_col.
    # -------------------------------------------------------------
    def determine_step_type(row):
        cval = row[charge_col]
        dval = row[discharge_col]
        if cval != 0 and abs(cval) > abs(dval):
            return "charge"
        elif dval != 0 and abs(dval) >= abs(cval):
            return "discharge"
        else:
            return "rest"

    df["step_type"] = df.apply(determine_step_type, axis=1)

    # -------------------------------------------------------------
    # 2) GROUPS: Mark transitions in step_type
    # -------------------------------------------------------------
    df["group_id"] = (df["step_type"] != df["step_type"].shift()).cumsum()

    # -------------------------------------------------------------
    # 3) FORWARD FILL zero values within each group
    #    after the first nonzero. This mimics your original
    #    selective_forward_fill logic but on step_type groups.
    # -------------------------------------------------------------
    def selective_forward_fill(group: pd.DataFrame) -> pd.DataFrame:
        # Identify first nonzero points, then forward fill subsequent zeros
        charge_mask = group[charge_col] != 0
        discharge_mask = group[discharge_col] != 0

        group[charge_col] = (
            group[charge_col]
            .where(charge_mask | (charge_mask.cumsum() == 0))
            .ffill()
        )
        group[discharge_col] = (
            group[discharge_col]
            .where(discharge_mask | (discharge_mask.cumsum() == 0))
            .ffill()
        )
        return group
    
    df = df.groupby("group_id", group_keys=False).apply(selective_forward_fill)

    # -------------------------------------------------------------
    # 4) Compute the combined capacity:
    #    combined = abs(charge_col) - abs(discharge_col)
    # -------------------------------------------------------------
    df[combined_col] = df[charge_col].abs() - df[discharge_col].abs()

    # -------------------------------------------------------------
    # 5) SHIFT & RESET LOGIC:
    #    - If the first group is discharge and min(capacity) < 0, shift up.
    #    - If a discharge group ends near (vmin + offset) and the next is charge,
    #      reset capacity to 0 at the start of the next group.
    # -------------------------------------------------------------

    # a) Find the global minimum voltage
    vmin = df[voltage_col].min()

    # We'll store final adjusted capacity in an additional column
    adj_col = combined_col + "_adj"
    df[adj_col] = np.nan

    # Keep track of offset from previous group (so capacity can remain continuous)
    capacity_offset = 0.0

    # Unique group IDs in ascending order
    group_ids = df["group_id"].unique()
    group_ids.sort()

    for i, gid in enumerate(group_ids):
        grp_mask = (df["group_id"] == gid)
        step_type = df.loc[grp_mask, "step_type"].iloc[0]

        # Extract the raw capacity in this group
        raw_cap = df.loc[grp_mask, combined_col].copy()

        if i == 0:
            # If the FIRST group is discharge and raw_cap < 0 anywhere, shift so min is 0
            if step_type == "discharge":
                min_val = raw_cap.min()
                if min_val < 0:
                    shift_amt = -min_val
                    raw_cap += shift_amt
            # Adjust by offset = 0 for the first group
            df.loc[grp_mask, adj_col] = raw_cap
            capacity_offset = raw_cap.iloc[-1]

        else:
            # Check the previous group's end conditions
            prev_gid = group_ids[i - 1]
            prev_mask = (df["group_id"] == prev_gid)

            prev_step_type = df.loc[prev_mask, "step_type"].iloc[0]
            prev_end_voltage = df.loc[prev_mask, voltage_col].iloc[-1]

            # If the previous group was discharge AND ended below (vmin + offset)
            # AND the current group is charge => reset capacity to 0
            if (
                prev_step_type == "discharge"
                and prev_end_voltage <= (vmin + min_voltage_offset)
                and step_type == "charge"
            ):
                capacity_offset = 0.0

            # Shift entire group by the current capacity_offset
            raw_cap_adj = raw_cap + capacity_offset

            df.loc[grp_mask, adj_col] = raw_cap_adj
            capacity_offset = raw_cap_adj.iloc[-1]

    # Rename the final adjusted column to your chosen combined_col, or keep both
    # Here, we'll overwrite combined_col with the adjusted values:
    df[combined_col] = df[adj_col]
    df.drop(columns=[adj_col], inplace=True)

    return df

def normalize_capacity(group_data, nominal_normalization, first_cycle_normalization, nominal_capacity):
    # Separate charge and discharge groups
    charge_groups = [entry for entry in group_data if entry['group_type'] == 'charge' and entry['crate'] != 0]
    discharge_groups = [entry for entry in group_data if entry['group_type'] == 'discharge' and entry['crate'] != 0]

    # Find the most common non-zero crate values for charge and discharge separately
    most_common_charge_crate = Counter([g['crate'] for g in charge_groups]).most_common(1)
    most_common_discharge_crate = Counter([g['crate'] for g in discharge_groups]).most_common(1)

    most_common_charge_crate = most_common_charge_crate[0][0] if most_common_charge_crate else None
    most_common_discharge_crate = most_common_discharge_crate[0][0] if most_common_discharge_crate else None

    # Find the first group with each most common crate and get its capacity
    first_charge_capacity = next((g['capacity'] for g in charge_groups if g['crate'] == most_common_charge_crate), None)
    first_discharge_capacity = next((g['capacity'] for g in discharge_groups if g['crate'] == most_common_discharge_crate), None)


    for g in group_data:
        if g['group_type'] not in ['charge', 'discharge']:
            continue
        # Label groups with the most common charge or discharge crate as "Cycle Aging"
        if g['group_type'] == 'charge':
            if g['crate'] == most_common_charge_crate:
                g['test_type'] = "Cycle Aging"
            else:
                g['test_type'] = "Rate Performance Test"
        

        if g['group_type'] == 'discharge':
            if g['crate'] == most_common_discharge_crate:
                g['test_type'] = "Cycle Aging"
            else:
                g['test_type'] = "Rate Performance Test"

        # Apply normalization
        if nominal_normalization:
            g['nominal_normalized_capacity'] = g['capacity'] / nominal_capacity if nominal_capacity else None
        if first_cycle_normalization:
            if g['group_type'] == 'charge' and first_charge_capacity:
                g['firstC_normalized_capacity'] = g['capacity'] / first_charge_capacity
            elif g['group_type'] == 'discharge' and first_discharge_capacity:
                g['firstC_normalized_capacity'] = g['capacity'] / first_discharge_capacity

    return group_data

def normalize_dcir(group_data, first_pulse, dcir_normalization):
    soc_limit = dcir_normalization[0]
    time_limit = dcir_normalization[1]

    for g in group_data:
        if g['pulse'] and g.get('soc') == soc_limit:
            key = f"internal_resistance_{time_limit}s"
            ir_val = g.get(key)
            if ir_val:
                g[f"normalized_{key}"] = ir_val / first_pulse
    return group_data




