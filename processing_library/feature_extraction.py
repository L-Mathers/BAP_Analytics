# feature_extraction.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def estimate_soc(
    df,
    voltage_col="voltage",
    step_col="Step Type",
    capacity_col="capacity",
    charge_capacity_col="charge_capacity",
    discharge_capacity_col="discharge_capacity",
    voltage_tolerance=0.1,
    update_nominal=False,
    nom_cap=None,
):
    """
    Estimate the State of Charge (SoC) for a battery test dataset, recalibrating
    whenever a 'last charge' (full) or 'last discharge' (empty) event is detected.

    Nominal capacity is updated at each 'last charge' event, using:
        new_nominal = df.loc[idx:, capacity_col].max()
    and that new nominal capacity is used *immediately* for subsequent rows.

    Args:
        df (pandas.DataFrame): The input DataFrame containing battery test data.
        voltage_col (str, optional): The column name for voltage. Defaults to 'voltage'.
        step_col (str, optional): The column name for step type. Defaults to 'Step Type'.
        capacity_col (str, optional): The column name for capacity. Defaults to 'capacity'.
        charge_capacity_col (str, optional): The column name for charge capacity. Defaults to 'charge_capacity'.
        discharge_capacity_col (str, optional): The column name for discharge capacity. Defaults to 'discharge_capacity'.
        voltage_tolerance (float, optional): The tolerance level for voltage comparison. Defaults to 0.1.
        update_nominal (bool, optional): Whether to update the nominal capacity. Defaults to False.
        nom_cap (float, optional): Nominal capacity value. Defaults to None.

    Returns a DataFrame with two new columns:
        'Accumulated Capacity' and 'Estimated SoC'.
    """
    # Check if charge and discharge capacity columns exist
    has_charge_cap = charge_capacity_col in df.columns
    has_discharge_cap = discharge_capacity_col in df.columns

    # Create a backup manual calculation for capacity differences
    df["time_diff"] = df["time"].diff().fillna(0)  # Calculate time differences
    df["capacity_manual"] = (
        df["current"] * df["time_diff"]
    ).cumsum() / 3600  # Integrate current over time to get capacity in Ah

    # Create a unified capacity difference column based on charge and discharge capacity
    if has_charge_cap and has_discharge_cap:
        # Calculate differences in charge and discharge capacity
        df["charge_cap_diff"] = df[charge_capacity_col].diff().fillna(0)
        df["discharge_cap_diff"] = df[discharge_capacity_col].diff().fillna(0)

        # Handle capacity resets by detecting when absolute capacity decreases
        # For charge capacity
        charge_reset_mask = df[charge_capacity_col].abs() < df[charge_capacity_col].shift(1).abs()
        df.loc[charge_reset_mask, "charge_cap_diff"] = df.loc[
            charge_reset_mask, charge_capacity_col
        ].abs()

        # For discharge capacity
        discharge_reset_mask = (
            df[discharge_capacity_col].abs() < df[discharge_capacity_col].shift(1).abs()
        )
        df.loc[discharge_reset_mask, "discharge_cap_diff"] = df.loc[
            discharge_reset_mask, discharge_capacity_col
        ].abs()

        # Create a unified capacity difference based on step type
        df["unified_cap_diff"] = 0.0

        # For charge steps, use charge capacity difference
        charge_mask = df[step_col].str.contains("charge", case=False, na=False)
        df.loc[charge_mask, "unified_cap_diff"] = df.loc[charge_mask, "charge_cap_diff"].abs()

        # For discharge steps, use discharge capacity difference
        discharge_mask = df[step_col].str.contains("discharge", case=False, na=False)
        df.loc[discharge_mask, "unified_cap_diff"] = df.loc[
            discharge_mask, "discharge_cap_diff"
        ].abs()

        # Use the unified capacity difference for cap_diff
        df["cap_diff"] = df["unified_cap_diff"]
    else:
        # Use the capacity column if charge/discharge capacity columns aren't available
        # First check if capacity column exists
        if capacity_col in df.columns:
            # Calculate capacity difference
            df["capacity_diff"] = df[capacity_col].diff().fillna(0)

            # Handle capacity resets by detecting when absolute capacity decreases
            capacity_reset_mask = df[capacity_col].abs() < df[capacity_col].shift(1).abs()
            df.loc[capacity_reset_mask, "capacity_diff"] = df.loc[
                capacity_reset_mask, capacity_col
            ].abs()

            # Use the capacity difference for cap_diff
            df["cap_diff"] = df["capacity_diff"].abs()
        else:
            # Fallback to manual calculation from current integration
            df["cap_diff"] = df["capacity_manual"].diff().abs().fillna(0)

    # Identify boundary voltages
    min_voltage = df[voltage_col].min()
    max_voltage = df[voltage_col].max()

    near_min = min_voltage + voltage_tolerance
    near_max = max_voltage - voltage_tolerance

    # Determine sign of each row: +1 if charge, -1 if discharge, 0 if rest/other
    step_type = df[step_col].astype(str).str.lower().fillna("")
    sign = np.where(
        step_type.str.contains("discharge"),
        -1,
        np.where(step_type.str.contains("charge"), 1, 0),
    )
    df["Sign"] = sign

    # Find "last charge" and "last discharge" events:
    #    last charge -> next step is rest, and voltage >= near_max
    #    last discharge -> next step is rest, and voltage <= near_min
    next_step_type = pd.Series(np.roll(step_type.values, -1), index=df.index)
    # Prevent potential edge confusion in the very last row:
    next_step_type.iloc[-1] = next_step_type.iloc[-1]

    is_last_charge_step = (
        (sign == 1) & (next_step_type.str.contains("rest")) & (df[voltage_col] >= near_max)
    )
    is_last_discharge_step = (
        (sign == -1) & (next_step_type.str.contains("rest")) & (df[voltage_col] <= near_min)
    )

    charge_indices = np.where(is_last_charge_step)[0]
    discharge_indices = np.where(is_last_discharge_step)[0]

    accumulated_capacity = np.zeros(len(df))
    estimated_soc = np.full(len(df), np.nan)

    # Combine the two event types into a single list, labeled, and sort by index
    events = []
    for idx in charge_indices:
        events.append((idx, "charge"))
    for idx in discharge_indices:
        events.append((idx, "discharge"))
    events.sort(key=lambda x: x[0])  # sort by the row index ascending

    # If no events, do a single pass cumsum with a single nominal.
    if len(events) == 0:
        # We'll just pick the largest capacity in the entire dataset as nominal:
        if update_nominal == False:
            nominal_capacity = nom_cap
        else:
            nominal_capacity = df[capacity_col].max()
        # Signed capacity changes
        raw_cap_diff = sign * df["cap_diff"].values
        accumulated_capacity[:] = np.cumsum(raw_cap_diff)
        estimated_soc[:] = (accumulated_capacity / nominal_capacity) * 100
        # Clip
        accumulated_capacity = np.clip(accumulated_capacity, 0, None)
        estimated_soc = np.clip(estimated_soc, 0, 100)
        df["Accumulated Capacity"] = accumulated_capacity
        df["soc"] = estimated_soc

        return df

    # We'll keep track of a "current" nominal capacity that can be updated
    # at each full-charge event.
    # Start with a nominal capacity guess (e.g., maximum in the entire dataset).
    if not update_nominal:
        nominal_capacity = nom_cap
    else:
        nominal_capacity = df[capacity_col].max()

    # Signed capacity changes for all rows
    raw_cap_diff = sign * df["cap_diff"].values

    # We'll accumulate segment-by-segment.
    start_idx = 0  # Start of the current segment
    current_baseline = 0.0  # Accumulated capacity at the last event

    # First partial segment: from row 0 up to (but not including) the first event
    first_event_idx = events[0][0]
    if first_event_idx > 0:
        # cumsum in [0, first_event_idx)
        segment_diffs = raw_cap_diff[start_idx:first_event_idx]

        # Calculate accumulated capacity for this segment
        segment_accumulated = current_baseline + np.cumsum(segment_diffs)

        # Check if we have negative values in this segment
        min_value = np.min(segment_accumulated)
        if min_value < 0:
            # Shift the entire segment up so the minimum value becomes zero
            shift_amount = abs(min_value)
            segment_accumulated += shift_amount

        # Assign the shifted values back to accumulated_capacity
        accumulated_capacity[start_idx:first_event_idx] = segment_accumulated

        estimated_soc[start_idx:first_event_idx] = (
            accumulated_capacity[start_idx:first_event_idx] / nominal_capacity
        ) * 100

        # No need for clipping since we've already ensured no negative values
        estimated_soc[start_idx:first_event_idx] = np.clip(
            estimated_soc[start_idx:first_event_idx], 0, 100
        )

    # Process each event in chronological order
    for i, (evt_idx, evt_type) in enumerate(events):
        # Fill cumsum from the previous event to this event index (exclusive)
        if i > 0:
            prev_event_idx = events[i - 1][0]
            segment_diffs = raw_cap_diff[prev_event_idx:evt_idx]

            # Calculate accumulated capacity for this segment
            segment_accumulated = current_baseline + np.cumsum(segment_diffs)

            # Check if we have negative values in this segment
            min_value = np.min(segment_accumulated)
            if min_value < 0:
                # Shift the entire segment up so the minimum value becomes zero
                shift_amount = abs(min_value)
                segment_accumulated += shift_amount

            # Assign the shifted values back to accumulated_capacity
            accumulated_capacity[prev_event_idx:evt_idx] = segment_accumulated

            estimated_soc[prev_event_idx:evt_idx] = (
                accumulated_capacity[prev_event_idx:evt_idx] / nominal_capacity
            ) * 100

            # No need for clipping since we've already ensured no negative values
            estimated_soc[prev_event_idx:evt_idx] = np.clip(
                estimated_soc[prev_event_idx:evt_idx], 0, 100
            )

        # Now handle the actual event row "evt_idx"
        if evt_type == "charge":
            # For full charge events, set SOC to 100%
            estimated_soc[evt_idx] = 100.0

            if update_nominal:
                new_nominal = df.loc[evt_idx:, capacity_col].max()
                nominal_capacity = new_nominal

                # Only update accumulated capacity if update_nominal is true
                accumulated_capacity[evt_idx] = nominal_capacity
                # Update the baseline for the next segment
                print("update entry WTF")

            # nominal_capacity = accumulated_capacity[evt_idx - 1]
            current_baseline = accumulated_capacity[evt_idx - 1]
            current_baseline = nominal_capacity
            print(f"Updated nominal capacity to {nominal_capacity} at index {evt_idx}")
            print(f"Current baseline set to {current_baseline} at index {evt_idx}")
            print("current accumulated capacity")

        elif evt_type == "discharge":
            # For full discharge events, set SOC to 0%
            estimated_soc[evt_idx] = 0.0

            accumulated_capacity[evt_idx] = 0.0
            current_baseline = 0.0  # Reset baseline for the next cycle

        # Move the segment start index to this event
        start_idx = evt_idx

    # After the last event, fill the cumsum to the end of the dataset
    last_event_idx = events[-1][0]
    if last_event_idx < len(df) - 1:
        segment_diffs = raw_cap_diff[last_event_idx:]

        # Calculate accumulated capacity for this segment
        segment_accumulated = current_baseline + np.cumsum(segment_diffs)

        # Check if we have negative values in this segment
        min_value = np.min(segment_accumulated)
        if min_value < 0:
            # Shift the entire segment up so the minimum value becomes zero
            shift_amount = abs(min_value)
            segment_accumulated += shift_amount

        # Assign the shifted values back to accumulated_capacity
        accumulated_capacity[last_event_idx:] = segment_accumulated

        estimated_soc[last_event_idx:] = (
            accumulated_capacity[last_event_idx:] / nominal_capacity
        ) * 100

    # Final check for any remaining negative values (should not happen with our shifting approach)
    min_accumulated = np.min(accumulated_capacity)
    if min_accumulated < 0:
        accumulated_capacity = np.clip(accumulated_capacity, 0, None)

    # Ensure SOC is within 0-100%
    estimated_soc = np.clip(estimated_soc, 0, 100)

    # Store in DataFrame
    df["Accumulated Capacity"] = accumulated_capacity
    df["soc"] = estimated_soc
    df.to_csv("PostSOC_estimation.csv", index=False)

    return df


def assign_cycle_keys(data):
    """
    Assigns cycle numbers if we have a charge->discharge pairing.
    If is_rpt=True, we only assign if the discharge is full_cycle.
    """
    last_charge = None
    cycle_number = 0
    for entry in data:
        if entry["group_type"] == "charge" and entry["test_type"] != "rpt":
            last_charge = entry
        elif entry["group_type"] == "discharge" and last_charge:
            if entry["test_type"] != "rpt":
                cycle_number += 1
                last_charge["cycle"] = cycle_number
                entry["cycle"] = cycle_number
                last_charge = None
            elif entry["test_type"] == "rpt":
                last_charge["cycle"] = cycle_number
                entry["cycle"] = cycle_number
    return data, cycle_number


def calculate_coulombic_and_energy_eff(group_data):
    """
    coulombic/energy efficiency calculations per (charge, discharge) pair.
    """

    # Identify distinct cycles
    cyc_map = {}
    for g in group_data:
        c = g.get("cycle")
        if c:
            cyc_map.setdefault(c, []).append(g)

    # For each cycle, if we have a charge and a discharge, compute efficiency
    for cnum, items in cyc_map.items():
        c_grp = next((x for x in items if x["group_type"] == "charge"), None)
        d_grp = next((x for x in items if x["group_type"] == "discharge"), None)
        if c_grp and d_grp:
            ch_cap = c_grp["capacity"]
            dch_cap = d_grp["capacity"]
            ch_en = c_grp["energy"]
            dch_en = d_grp["energy"]

            if ch_cap and dch_cap and ch_cap != 0:
                eff = (dch_cap / ch_cap) * 100
                c_grp["coulombic_efficiency"] = eff
                d_grp["coulombic_efficiency"] = eff
            if ch_en and dch_en and ch_en != 0:
                e_eff = (dch_en / ch_en) * 100
                c_grp["energy_efficiency"] = e_eff
                d_grp["energy_efficiency"] = e_eff

    return group_data


# data_processing.py

from collections import Counter

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
        start_time=("time", "first"), end_time=("time", "last")
    )
    group_durations["duration"] = group_durations["end_time"] - group_durations["start_time"]
    valid = group_durations[group_durations["duration"] > 120].index

    # Filter only for averaging purposes, not df output
    filtered = df[df["group"].isin(valid)]

    df.to_csv("PostGroup_assignment.csv")
    # Approx pseudo-limits from the filtered region
    avg_last_volt_charge = (
        filtered[filtered["Step Type"] == "charge"].groupby("group")["voltage"].last().mean()
    )
    avg_last_volt_discharge = (
        filtered[filtered["Step Type"] == "discharge"].groupby("group")["voltage"].last().mean()
    )

    return (
        df,
        round(avg_last_volt_charge, 4) if pd.notnull(avg_last_volt_charge) else vmax,
        round(avg_last_volt_discharge, 4) if pd.notnull(avg_last_volt_discharge) else vmin,
    )


def add_cv_steps(
    df: pd.DataFrame,
    vhigh: float,
    vlow: float,
    voltage_tolerance: float = 1e-2,
    current_delta_threshold: float = 0.01,
    voltage_delta_threshold: float = 0.0001,
):
    """
    This function modifies the DataFrame by identifying 'constant voltage (CV)' phases
    during 'charge' and 'discharge' steps. It updates the 'Step Type' column to reflect
    CV once a threshold condition is met. Sequences < 10 steps are reverted.
    """
    # Ensure the DataFrame contains the necessary columns
    required_columns = ["Step Type", "voltage", "current", "group"]
    if not all(col in df.columns for col in required_columns):
        print("Error: Required columns are missing in the DataFrame.")
        return df

    df["delta_current"] = df["current"].diff().abs()
    df["delta_voltage"] = df["voltage"].diff().abs()

    # Process each group separately
    for group_number, group_data in df.groupby("group"):
        step_type = group_data["Step Type"].iloc[0]
        # Exclude the first 10 rows
        group_data = group_data.iloc[10:]

        if step_type == "charge":
            # Condition for 'charge cv'
            cv_condition = (group_data["delta_current"] >= current_delta_threshold) & (
                group_data["delta_voltage"] <= voltage_delta_threshold
            )
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, "Step Type"] = "charge cv"

        elif step_type == "discharge":
            # Condition for 'discharge cv'
            cv_condition = group_data["delta_current"] >= current_delta_threshold
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, "Step Type"] = "discharge cv"

    # Remove CV sequences that are less than 10 consecutive steps
    df["cv_flag"] = (df["Step Type"].str.contains("cv")).fillna(False).astype(int)
    df["cv_group"] = (df["cv_flag"] != df["cv_flag"].shift()).cumsum()

    for cv_group, group_data in df.groupby("cv_group"):
        if group_data["cv_flag"].iloc[0] == 1 and len(group_data) < 10:
            # Revert to original step type
            # If we want to guess "charge" or "discharge":
            original_type = (
                "discharge" if "discharge cv" in group_data["Step Type"].unique()[0] else "charge"
            )
            df.loc[group_data.index, "Step Type"] = original_type
    df.to_csv("PostCCCV_Assignmnet.csv")
    df.drop(columns=["delta_current", "cv_flag", "cv_group", "delta_voltage"], inplace=True)
    return df


def match_required_columns(data: pd.DataFrame, threshold: int = 80):
    """
    Fuzzy-match required columns in a DataFrame.
    If a match fails, raises ValueError.
    Returns a dict suitable for `data.rename(columns=...)`.
    """
    from fuzzywuzzy import fuzz, process

    required_cols = {
        "time": ["time", "elapsed time", "test time", "time (s)"],
        "current": ["current", "current (a)"],
        "voltage": ["voltage", "voltage (v)"],
        "discharge_capacity": ["discharge capacity (ah)", "dcap", "discharge_capacity"],
        "charge_capacity": ["charge capacity (ah)", "ccap", "charge_capacity"],
        "discharge_energy": ["discharge energy (wh)", "denergy", "discharge_energy"],
        "charge_energy": ["charge energy (wh)", "cenergy", "charge_energy"],
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
    charge_col="charge_capacity",
    discharge_col="discharge_capacity",
    voltage_col="voltage",
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
            group[charge_col].where(charge_mask | (charge_mask.cumsum() == 0)).ffill()
        )
        group[discharge_col] = (
            group[discharge_col].where(discharge_mask | (discharge_mask.cumsum() == 0)).ffill()
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
        grp_mask = df["group_id"] == gid
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
            prev_mask = df["group_id"] == prev_gid

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


def normalize_capacity(
    group_data, nominal_normalization, first_cycle_normalization, nominal_capacity
):
    """
    Normalize capacity based on nominal capacity or first cycle capacity.

    Args:
        group_data (list): List of group dictionaries
        nominal_normalization (bool): Whether to normalize by nominal capacity
        first_cycle_normalization (bool): Whether to normalize by first cycle capacity
        nominal_capacity (float): Nominal capacity value

    Returns:
        list: Updated group_data with normalized capacity values
    """
    if not (nominal_normalization or first_cycle_normalization):
        return group_data

    # Organize cycle data
    cycle_data = {}
    total_charge, total_discharge, count_charge, count_discharge = 0, 0, 0, 0

    for g in group_data:
        if g["test_type"] != "cycling":
            continue
        cycle = g.get("cycle", 0)
        if cycle <= 0:
            continue

        if cycle not in cycle_data:
            cycle_data[cycle] = {"charge": None, "discharge": None}

        capacity = g.get("capacity", 0)
        if g["group_type"] == "charge" and capacity > 0:
            cycle_data[cycle]["charge"] = capacity
            total_charge += capacity
            count_charge += 1
        elif g["group_type"] == "discharge" and capacity > 0:
            cycle_data[cycle]["discharge"] = capacity
            total_discharge += capacity
            count_discharge += 1

    # Calculate averages
    avg_charge_capacity = total_charge / count_charge if count_charge else 0
    avg_discharge_capacity = total_discharge / count_discharge if count_discharge else 0

    # Find suitable first complete cycle (>= 90% avg)
    first_charge_capacity, first_discharge_capacity = None, None

    for cycle, data in sorted(cycle_data.items()):
        charge_valid = data["charge"] and data["charge"] >= 0.9 * avg_charge_capacity
        discharge_valid = data["discharge"] and data["discharge"] >= 0.9 * avg_discharge_capacity

        if charge_valid and discharge_valid:
            first_charge_capacity = data["charge"]
            first_discharge_capacity = data["discharge"]
            print(
                f"Found first complete cycle: {cycle} with charge capacity {first_charge_capacity} and discharge capacity {first_discharge_capacity}"
            )
            break

    if not (first_charge_capacity and first_discharge_capacity):
        print("Warning: No suitable first complete cycle found for normalization")
        return group_data

    # Apply normalization
    for g in group_data:
        if nominal_normalization and nominal_capacity > 0 and g.get("capacity", 0) is not None:
            g["nominal_normalized_capacity"] = (g["capacity"] / nominal_capacity) * 100

        if first_cycle_normalization and g.get("capacity", 0) is not None:
            if g["group_type"] == "charge" and first_charge_capacity > 0:
                g["first_cycle_normalized_capacity"] = (g["capacity"] / first_charge_capacity) * 100
            elif g["group_type"] == "discharge" and first_discharge_capacity > 0:
                g["first_cycle_normalized_capacity"] = (
                    g["capacity"] / first_discharge_capacity
                ) * 100

    # Count how many groups were normalized
    nominal_count = sum(1 for g in group_data if "nominal_normalized_capacity" in g)
    first_cycle_count = sum(1 for g in group_data if "first_cycle_normalized_capacity" in g)

    if nominal_normalization:
        print(f"Applied nominal normalization to {nominal_count} groups")
    if first_cycle_normalization:
        print(f"Applied first cycle normalization to {first_cycle_count} groups")

    return group_data


def normalize_dcir(group_data, first_pulse, dcir_normalization):
    """
    Normalize internal resistance values by dividing by the first pulse value.

    Args:
        group_data (list): List of group dictionaries
        first_pulse (float): Reference resistance value to normalize against
        dcir_normalization (tuple): (SOC limit, time limit) used to identify reference cells

    Returns:
        list: Updated group_data with normalized resistance values added
    """
    # Validate dcir_normalization
    if not dcir_normalization or len(dcir_normalization) != 2:
        print(
            f"Warning: Invalid dcir_normalization format: {dcir_normalization}. Skipping normalization."
        )
        return group_data

    soc_limit = dcir_normalization[0]
    time_limit = dcir_normalization[1]

    # First ensure first_pulse is valid
    if not first_pulse or first_pulse <= 0:
        print("Warning: Invalid first_pulse value for normalization:", first_pulse)
        return group_data

    # Add normalized values to all matching groups
    normalized_key = f"normalized_internal_resistance_{time_limit}s"
    key = f"internal_resistance_{time_limit}s"

    for g in group_data:
        # Only normalize DCIR values from pulse measurements
        if g["pulse"] and key in g:
            ir_val = g.get(key)
            if ir_val and ir_val > 0:
                # Add the normalized value to the same group
                g[normalized_key] = ir_val / first_pulse
    return group_data


def seperate_test_types(group_data, test_type, tolerance=0.05):
    """
    Classifies groups as either 'cycling' or 'rpt' based on their C-rates.

    Args:
        group_data (list): List of dictionaries containing group information
        test_type (str): The test type ("Rate Performance Test", "Cycle Aging", or "Combined RPT/Cycling")
        tolerance (float): C-rate tolerance for determining if a group matches the average

    Returns:
        list: The updated group_data with 'test_type' keys added
    """
    # Separate charge and discharge groups with non-zero C-rates
    charge_groups = [g for g in group_data if g["group_type"] == "charge" and g.get("crate", 0) > 0]
    discharge_groups = [
        g for g in group_data if g["group_type"] == "discharge" and g.get("crate", 0) > 0
    ]

    # Calculate average C-rates
    avg_charge_crate = (
        sum(g.get("crate", 0) for g in charge_groups) / len(charge_groups) if charge_groups else 0
    )
    avg_discharge_crate = (
        sum(g.get("crate", 0) for g in discharge_groups) / len(discharge_groups)
        if discharge_groups
        else 0
    )

    # Assign test_type to each group
    previous_test_type = None

    # Handle the different test types consistently
    if test_type == "Rate Performance Test":
        for g in group_data:
            g["test_type"] = "rpt"
    elif test_type == "Cycle Aging":
        for g in group_data:
            g["test_type"] = "cycling"
    elif test_type == "Combined RPT/Cycling":
        for g in group_data:
            if g["group_type"] == "charge":
                crate = g.get("crate", 0)
                if abs(crate - avg_charge_crate) <= tolerance and not g["pulse"]:
                    g["test_type"] = "cycling"
                    previous_test_type = "cycling"
                else:
                    g["test_type"] = "rpt"
                    previous_test_type = "rpt"
            elif g["group_type"] == "discharge":
                crate = g.get("crate", 0)
                if abs(crate - avg_discharge_crate) <= tolerance and not g["pulse"]:
                    g["test_type"] = "cycling"
                    previous_test_type = "cycling"
                else:
                    g["test_type"] = "rpt"
                    previous_test_type = "rpt"
            else:
                g["test_type"] = previous_test_type if previous_test_type else "unknown"
    else:
        # Default behavior for unknown test types
        for g in group_data:
            g["test_type"] = "unknown"
    cycling_count = 0
    rpt_count = 0
    for g in group_data:
        if g["test_type"] == "cycling":
            cycling_count += 1
        elif g["test_type"] == "rpt":
            rpt_count += 1

    print(f"found {cycling_count} cycling groups and {rpt_count} rpt groups")

    return group_data
