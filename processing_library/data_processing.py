# data_processing.py

import numpy as np
import pandas as pd


def psuedo_limit_extraction(df: pd.DataFrame, zero_current_tolerance: float = 0.1):
    """
    Identifies rest/charge/discharge in the DataFrame using a current threshold.
    Removes short groups (<10 points) and returns pseudo-limits.
    """
    if "current" not in df.columns or "voltage" not in df.columns:
        raise ValueError("Error: Required columns 'current' and 'voltage' not found.")

    current_abs = df["current"].abs()
    current_value = df["current"].where(current_abs > zero_current_tolerance, 0)
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
            # If we want to guess “charge” or “discharge”:
            original_type = 'charge' if 'charge cv' in group_data['Step Type'].unique()[0] else 'discharge'
            df.loc[group_data.index, 'Step Type'] = original_type

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