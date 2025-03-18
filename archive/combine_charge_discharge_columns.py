import pandas as pd
import numpy as np


def combine_charge_discharge_columns(
    df: pd.DataFrame,
    translations: dict,
    target: str,
    voltage_tolerance=0.1,
    zero_current_tolerance: float = 0.1,
) -> pd.DataFrame:
    """
    Identifies rest/charge/discharge in the DataFrame using a current threshold.
    Removes short groups (<10 points) and computes accumulated capacity.
    """
    df = df.copy(deep=True)

    # Get current and voltage columns
    current_col = translations["Current (A)"]
    voltage_col = translations["Voltage (V)"]

    if not current_col and not voltage_col:
        raise ValueError("Error: Required columns 'current' and 'voltage' not found.")

    # Get charge, discharge, and combined columns
    if target == "Capacity (Ah)":
        charge_col = translations["Charge Capacity (Ah)"]
        discharge_col = translations["Discharge Capacity (Ah)"]
        combined_col = translations["Capacity (Ah)"]
    elif target == "Energy (Wh)":
        charge_col = translations["Charge Energy (Wh)"]
        discharge_col = translations["Discharge Energy (Wh)"]
        combined_col = translations["Energy (Wh)"]
    else:
        raise ValueError(
            "Error: Invalid target value. Must be 'Capacity (Ah)' or 'Energy (Wh)'."
        )

    combined_col_name = target

    current_abs = df[current_col].abs()
    current_value = df[current_col].where(current_abs > zero_current_tolerance, 0)

    df["current"] = current_value
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

    step_col = "Step Type"
    
    # Calculate reset indices early for both functions to use
    min_voltage = df[voltage_col].min()
    max_voltage = df[voltage_col].max()

    near_min = min_voltage + voltage_tolerance
    near_max = max_voltage - voltage_tolerance

    step_type = df[step_col].astype(str).str.lower().fillna("")
    sign = np.where(
        step_type.str.contains("discharge"),
        -1,
        np.where(step_type.str.contains("charge"), 1, 0),
    )
    df["Sign"] = sign

    next_step_type = pd.Series(np.roll(step_type.values, -1), index=df.index)
    next_step_type.iloc[-1] = next_step_type.iloc[-1]

    is_last_charge_step = (
        (sign == 1)
        & (next_step_type.str.contains("rest"))
        & (df[voltage_col] >= near_max)
    )
    is_last_discharge_step = (
        (sign == -1)
        & (next_step_type.str.contains("rest"))
        & (df[voltage_col] <= near_min)
    )

    charge_reset_indices = np.where(is_last_charge_step)[0]
    discharge_reset_indices = np.where(is_last_discharge_step)[0]
    
    if charge_col in df.columns and discharge_col in df.columns:
        # Check if columns need to be made continuous
        charge_zeros = (df[charge_col] == 0).sum()
        charge_nonzeros = len(df) - charge_zeros
        discharge_zeros = (df[discharge_col] == 0).sum()
        discharge_nonzeros = len(df) - discharge_zeros

        needs_continuous_charge = charge_zeros > charge_nonzeros
        needs_continuous_discharge = discharge_zeros > discharge_nonzeros

        if needs_continuous_charge or needs_continuous_discharge:
            df = make_charge_discharge_continuous(
                df, 
                charge_col, 
                discharge_col, 
                step_col, 
                charge_reset_indices, 
                discharge_reset_indices
            )

        # Capacity deltas
        df['charge_cap_diff'] = df[charge_col].diff().fillna(0)
        df['discharge_cap_diff'] = df[discharge_col].diff().fillna(0)

        charge_reset_mask = df[charge_col].abs() < df[charge_col].shift(1).abs()
        df.loc[charge_reset_mask, 'charge_cap_diff'] = df.loc[charge_reset_mask, charge_col].abs()

        discharge_reset_mask = df[discharge_col].abs() < df[discharge_col].shift(1).abs()
        df.loc[discharge_reset_mask, 'discharge_cap_diff'] = df.loc[discharge_reset_mask, discharge_col].abs()

        df['unified_cap_diff'] = 0.0
        charge_mask = df[step_col].str.contains('charge', case=False, na=False)
        discharge_mask = df[step_col].str.contains('discharge', case=False, na=False)

        df.loc[charge_mask, 'unified_cap_diff'] = df.loc[charge_mask, 'charge_cap_diff'].abs()
        df.loc[discharge_mask, 'unified_cap_diff'] = df.loc[discharge_mask, 'discharge_cap_diff'].abs()
        df["cap_diff"] = df['unified_cap_diff']

        # Drop temp columns
        df.drop(
            columns=["charge_cap_diff", "discharge_cap_diff", "unified_cap_diff"],
            inplace=True,
        )

    elif combined_col in df.columns:
        # Calculate capacity difference
        df["capacity_diff"] = df[combined_col].diff().fillna(0)

        # Handle capacity resets by detecting when absolute capacity decreases
        capacity_reset_mask = df[combined_col].abs() < df[combined_col].shift(1).abs()
        df.loc[capacity_reset_mask, "capacity_diff"] = df.loc[
            capacity_reset_mask, combined_col
        ].abs()

        # Use the capacity difference for cap_diff
        df["cap_diff"] = df["capacity_diff"].abs()
    else:
        df["time_diff"] = df["time"].diff().fillna(0)  # Calculate time differences
        df["cap_diff"] = (df[current_col] * df["time_diff"]).cumsum() / 3600
        df.drop(columns=["time_diff"], inplace=True)

    charge_indices = charge_reset_indices
    discharge_indices = discharge_reset_indices

    accumulated_capacity = np.zeros(len(df))

    events = []
    for idx in charge_indices:
        events.append((idx, "charge"))
    for idx in discharge_indices:
        events.append((idx, "discharge"))
    events.sort(key=lambda x: x[0])

    raw_cap_diff = sign * df["cap_diff"].values
    start_idx = 0
    current_baseline = 0.0

    if len(events) == 0:
        accumulated_capacity[:] = np.cumsum(raw_cap_diff)
        accumulated_capacity = np.clip(accumulated_capacity, 0, None)
        df[combined_col] = accumulated_capacity
        return df

    first_event_idx = events[0][0]
    if first_event_idx > 0:
        segment_diffs = raw_cap_diff[start_idx:first_event_idx]
        segment_accumulated = current_baseline + np.cumsum(segment_diffs)
        min_value = np.min(segment_accumulated)
        if min_value < 0:
            segment_accumulated += abs(min_value)
        accumulated_capacity[start_idx:first_event_idx] = segment_accumulated

    for i, (evt_idx, evt_type) in enumerate(events):
        if i > 0:
            prev_event_idx = events[i - 1][0]
            segment_diffs = raw_cap_diff[prev_event_idx:evt_idx]
            segment_accumulated = current_baseline + np.cumsum(segment_diffs)
            print(f"DEBUG: Segment {prev_event_idx} to {evt_idx}, baseline={current_baseline:.4f}, min={np.min(segment_accumulated):.4f}, max={np.max(segment_accumulated):.4f}")
            min_value = np.min(segment_accumulated)
            if min_value < 0:
                segment_accumulated += abs(min_value)
                print(f"DEBUG: Adjusted negative values by adding {abs(min_value):.4f}")
            if len(segment_accumulated) > 0:
                print(f"DEBUG: First 3 values: {segment_accumulated[:min(3, len(segment_accumulated))]}")
                print(f"DEBUG: Last 3 values: {segment_accumulated[-min(3, len(segment_accumulated)):]}")
                
            # Important change: Don't overwrite the previous event point (which might be a discharge)
            # Skip the first point of segment_accumulated and the first index of the range
            if len(segment_accumulated) > 1:  # Only if there are at least 2 points
                accumulated_capacity[prev_event_idx+1:evt_idx] = segment_accumulated[1:]
                print(f"DEBUG: Preserved event point at {prev_event_idx}, value now = {accumulated_capacity[prev_event_idx]:.4f}")
            
            # Check if this is a discharge point
            if events[i-1][1] == "discharge":
                print(f"DEBUG: After segment calc, discharge point at {prev_event_idx} value = {accumulated_capacity[prev_event_idx]:.4f}")
                # Check if any subsequent operations modified this value
                prev_discharge_val = accumulated_capacity[prev_event_idx]

        # Set the current event point based on the event type
        if evt_type == "charge":
            print(f"DEBUG: Charge event at index {evt_idx}")
            print(f"DEBUG: Before setting, accumulated_capacity[{evt_idx}] = {accumulated_capacity[evt_idx]:.4f}")
            # For charge events, use the previous point's value
            if evt_idx > 0:
                print(f"DEBUG: Setting to accumulated_capacity[{evt_idx-1}] = {accumulated_capacity[evt_idx-1]:.4f}")
                accumulated_capacity[evt_idx] = accumulated_capacity[evt_idx - 1]
                current_baseline = accumulated_capacity[evt_idx - 1]
            else:
                # If this is the very first point, keep it at zero
                current_baseline = 0.0
            print(f"DEBUG: After setting, accumulated_capacity[{evt_idx}] = {accumulated_capacity[evt_idx]:.4f}")

        elif evt_type == "discharge":
            print(f"DEBUG: Discharge event at index {evt_idx}")
            print(f"DEBUG: Before any changes, accumulated_capacity[{evt_idx}] = {accumulated_capacity[evt_idx]:.4f}")
            
            # For discharge events, also use the previous point's value rather than keeping zero
            if evt_idx > 0:
                discharge_val = accumulated_capacity[evt_idx - 1]
                print(f"DEBUG: Setting discharge point to previous value: {discharge_val:.4f}")
                accumulated_capacity[evt_idx] = discharge_val
            else:
                discharge_val = 0.0
            
            # Keep the discharge event value, set the next index to zero instead
            # Check if there's a valid next index within the array bounds
            if evt_idx + 1 < len(accumulated_capacity):
                print(f"DEBUG: Setting index {evt_idx + 1} to zero after discharge event")
                accumulated_capacity[evt_idx + 1] = 0.0
                # Set the current_baseline for the next segment calculation
                if i < len(events) - 1 and events[i + 1][0] > evt_idx + 1:
                    print(f"DEBUG: Next event is at index {events[i + 1][0]}, setting baseline to 0")
                    current_baseline = 0.0
                else:
                    # If the next event is at the next index, keep the current baseline
                    print(f"DEBUG: No event or event at index {evt_idx + 1}, keeping baseline = {accumulated_capacity[evt_idx]:.4f}")
                    current_baseline = accumulated_capacity[evt_idx]
            else:
                # If we're at the last index, we can't set the next one to zero
                print(f"DEBUG: Discharge event at last index, can't set next to zero")
                # but we should still reset the baseline for consistency
                current_baseline = 0.0
                
            # Verify discharge value hasn't changed
            if discharge_val != accumulated_capacity[evt_idx]:
                print(f"DEBUG: WARNING! Discharge value changed from {discharge_val:.4f} to {accumulated_capacity[evt_idx]:.4f}")

        start_idx = evt_idx

    last_event_idx = events[-1][0]
    last_event_type = events[-1][1]
    print(f"DEBUG: Last event: type={last_event_type}, idx={last_event_idx}")
    
    if last_event_type == "discharge":
        print(f"DEBUG: Before final segment, discharge value at {last_event_idx} = {accumulated_capacity[last_event_idx]:.4f}")
    
    if last_event_idx < len(df) - 1:
        segment_diffs = raw_cap_diff[last_event_idx:]
        segment_accumulated = current_baseline + np.cumsum(segment_diffs)
        print(f"DEBUG: Final segment {last_event_idx} to end, baseline={current_baseline:.4f}")
        min_value = np.min(segment_accumulated)
        if min_value < 0:
            segment_accumulated += abs(min_value)
            print(f"DEBUG: Adjusted negative values in final segment by adding {abs(min_value):.4f}")
        
        # Same fix for final segment: Don't overwrite the last event point
        if len(segment_accumulated) > 1:
            accumulated_capacity[last_event_idx+1:] = segment_accumulated[1:]
            print(f"DEBUG: Preserved last event point, value = {accumulated_capacity[last_event_idx]:.4f}")
        
        if last_event_type == "discharge":
            # The issue may be here - we might be overwriting the discharge value
            print(f"DEBUG: After final segment, discharge value at {last_event_idx} = {accumulated_capacity[last_event_idx]:.4f}")
            # Check if we're incorrectly setting this to zero or calculating it wrong
            if accumulated_capacity[last_event_idx] == 0.0:
                print(f"DEBUG: ERROR: Last discharge value is zero!")
                print(f"DEBUG: Raw diff at this point = {raw_cap_diff[last_event_idx]:.4f}")
                print(f"DEBUG: segment_accumulated[0] = {segment_accumulated[0]:.4f}")

    min_accumulated = np.min(accumulated_capacity)
    if min_accumulated < 0:
        print(f"DEBUG: Clipping negative values from {min_accumulated:.4f} to 0")
        accumulated_capacity = np.clip(accumulated_capacity, 0, None)
        
    # Final check for discharge events
    for idx, evt_type in events:
        if evt_type == "discharge":
            print(f"DEBUG: Final check - Discharge at {idx}: value = {accumulated_capacity[idx]:.4f}")
            if accumulated_capacity[idx] == 0.0:
                print(f"DEBUG: WARNING! Discharge event at {idx} has zero value in final output!")

    columns_to_drop = ["current", "Step Type", "group", "cap_diff", "Sign"]

    df.drop(columns=columns_to_drop, inplace=True)

    df[combined_col_name] = accumulated_capacity

    df.to_csv("/Users/liammathers/Desktop/Github/bmw_lifetime_processing_int/temp_data/combined_charge_discharge.csv", index=False)
    return df


def make_charge_discharge_continuous(
    df, 
    charge_col, 
    discharge_col, 
    step_col, 
    charge_reset_indices, 
    discharge_reset_indices
):
    """
    Make charge and discharge capacity columns continuous by filling in gaps 
    based on step type and the previous non-rest step type.
    Only processes the column corresponding to the current step type.
    
    Args:
        df: DataFrame with charge and discharge data
        charge_col: Name of charge capacity column
        discharge_col: Name of discharge capacity column
        step_col: Name of step type column
        charge_reset_indices: Not used in current logic, kept for function signature
        discharge_reset_indices: Not used in current logic, kept for function signature
    
    Returns:
        DataFrame with continuous charge and discharge columns
    """
    df = df.copy()
    # Get all groups by number for easier lookup
    groups_by_num = {}
    for group_num, group in df.groupby('group'):
        groups_by_num[group_num] = {
            'indices': group.index,
            'step_type': group[step_col].iloc[0].lower(),
            'start_idx': group.index[0],
            'end_idx': group.index[-1]
        }
    first_non_rest_found = False
    # Find previous non-rest step type for each group
    for group_num in sorted(groups_by_num.keys()):
        current_group = groups_by_num[group_num]
        current_step_type = current_group['step_type']
        
        # Find previous non-rest group
        prev_non_rest_type = None
        for prev_num in range(group_num-1, 0, -1):
            if prev_num in groups_by_num and groups_by_num[prev_num]['step_type'] != 'rest':
                prev_non_rest_type = groups_by_num[prev_num]['step_type']
                break
        # Get the group data
        group_indices = current_group['indices']
        start_idx = current_group['start_idx']
        end_idx = current_group['end_idx']
        
        # Process charge column if this is a charge step
        if current_step_type == 'charge':
            # Get non-zero values and their indices
            nonzero_mask = df.loc[group_indices, charge_col] != 0
            nonzero_values = df.loc[group_indices[nonzero_mask], charge_col]
            nonzero_count = len(nonzero_values)
            zero_count = len(group_indices) - nonzero_count
        
            # Determine whether to reset based on previous non-rest step type
            should_reset = prev_non_rest_type != 'charge' and prev_non_rest_type is not None
            if len(nonzero_values) > 0:
                nonzero_indices = nonzero_values.index
                
                if should_reset or not first_non_rest_found:
                    first_non_rest_found = True
                    # Interpolate from zero to non-zero values
                    if len(nonzero_indices) > 0:
                        first_nonzero_idx = nonzero_indices[0]
                        last_nonzero_idx = nonzero_indices[-1]
                        first_nonzero_val = nonzero_values.iloc[0]
                        
                        # Create interpolation indices (from start_idx to first_nonzero_idx)
                        interp_range = pd.Index(range(start_idx, first_nonzero_idx + 1))
                        if len(interp_range) > 1:
                            # Linear interpolation from 0 to first_nonzero_val
                            interp_values = np.linspace(0, first_nonzero_val, len(interp_range))
                            df.loc[interp_range, charge_col] = interp_values
                        # Fill any gaps between non-zero values
                        if len(nonzero_indices) > 1:
                            for i in range(len(nonzero_indices) - 1):
                                curr_idx = nonzero_indices[i]
                                next_idx = nonzero_indices[i+1]
                                if next_idx - curr_idx > 1:
                                    # Indices between current and next non-zero values
                                    gap_indices = pd.Index(range(curr_idx + 1, next_idx))
                                    curr_val = df.loc[curr_idx, charge_col]
                                    next_val = df.loc[next_idx, charge_col]
                                    # Linear interpolation between values
                                    interp_values = np.linspace(curr_val, next_val, len(gap_indices) + 2)[1:-1]
                                    df.loc[gap_indices, charge_col] = interp_values
                        # Fill any trailing gaps after the last non-zero value
                        if last_nonzero_idx < end_idx:
                            trailing_indices = pd.Index(range(last_nonzero_idx + 1, end_idx + 1))
                            last_val = df.loc[last_nonzero_idx, charge_col]
                            df.loc[trailing_indices, charge_col] = last_val
                            print(f"DEBUG: Filled trailing {len(trailing_indices)} values with {last_val}")
                            if len(trailing_indices) <= 5:
                                print(f"DEBUG: Trailing values are now: {df.loc[trailing_indices, charge_col].values}")

                    # Print full group values after all interpolations
                else:
                    # Use the last value from previous group as baseline
                    if start_idx > 0:
                        prev_val = df.loc[start_idx - 1, charge_col]                        
                        # Create series of values starting from prev_val and incorporating non-zero changes
                        curr_val = prev_val
                        updates = 0
                        for idx in group_indices:
                            if idx in nonzero_indices:
                                # Update current value only at non-zero points
                                old_val = curr_val
                                curr_val = df.loc[idx, charge_col]
                                updates += 1
                            df.loc[idx, charge_col] = curr_val
             
            prev_discharge_val = df.loc[start_idx - 1, discharge_col]
            df.loc[group_indices, discharge_col] = prev_discharge_val
        # Process discharge column if this is a discharge step
        elif current_step_type == 'discharge':
            # Get non-zero values and their indices
            nonzero_mask = df.loc[group_indices, discharge_col] != 0
            nonzero_values = df.loc[group_indices[nonzero_mask], discharge_col]
            nonzero_count = len(nonzero_values)
            zero_count = len(group_indices) - nonzero_count
            # Determine whether to reset based on previous non-rest step type
            should_reset = prev_non_rest_type != 'discharge' and prev_non_rest_type is not None            
            if len(nonzero_values) > 0:
                nonzero_indices = nonzero_values.index
                
                if should_reset or not first_non_rest_found:
                    first_non_rest_found = True
                    # Interpolate from zero to non-zero values
                    if len(nonzero_indices) > 0:
                        first_nonzero_idx = nonzero_indices[0]
                        last_nonzero_idx = nonzero_indices[-1]
                        first_nonzero_val = nonzero_values.iloc[0]
                        
                       
                        # Create interpolation indices (from start_idx to first_nonzero_idx)
                        interp_range = pd.Index(range(start_idx, first_nonzero_idx + 1))
                        if len(interp_range) > 1:
                            # Linear interpolation from 0 to first_nonzero_val
                            interp_values = np.linspace(0, first_nonzero_val, len(interp_range))
                            df.loc[interp_range, discharge_col] = interp_values
                           
                        # Fill any gaps between non-zero values
                        if len(nonzero_indices) > 1:
                            for i in range(len(nonzero_indices) - 1):
                                curr_idx = nonzero_indices[i]
                                next_idx = nonzero_indices[i+1]
                                if next_idx - curr_idx > 1:
                                    # Indices between current and next non-zero values
                                    gap_indices = pd.Index(range(curr_idx + 1, next_idx))
                                    curr_val = df.loc[curr_idx, discharge_col]
                                    next_val = df.loc[next_idx, discharge_col]
                                    # Linear interpolation between values
                                    interp_values = np.linspace(curr_val, next_val, len(gap_indices) + 2)[1:-1]
                                    df.loc[gap_indices, discharge_col] = interp_values
                                   
                        # Fill any trailing gaps after the last non-zero value
                        if last_nonzero_idx < end_idx:
                            trailing_indices = pd.Index(range(last_nonzero_idx + 1, end_idx + 1))
                            last_val = df.loc[last_nonzero_idx, discharge_col]
                            df.loc[trailing_indices, discharge_col] = last_val
                           
                else:
                    # Use the last value from previous group as baseline
                    if start_idx > 0:
                        prev_val = df.loc[start_idx - 1, discharge_col]
                        
                        # Create series of values starting from prev_val and incorporating non-zero changes
                        curr_val = prev_val
                        updates = 0
                        for idx in group_indices:
                            if idx in nonzero_indices:
                                # Update current value only at non-zero points
                                old_val = curr_val
                                curr_val = df.loc[idx, discharge_col]
                                updates += 1
                            df.loc[idx, discharge_col] = curr_val
              
            prev_charge_val = df.loc[start_idx - 1, charge_col]
            df.loc[group_indices, charge_col] = prev_charge_val
        # For rest periods, maintain the last value from previous step
        elif current_step_type == 'rest':
            if start_idx > 0:
                prev_charge_val = df.loc[start_idx - 1, charge_col]
                prev_discharge_val = df.loc[start_idx - 1, discharge_col]
                df.loc[group_indices, charge_col] = prev_charge_val
                df.loc[group_indices, discharge_col] = prev_discharge_val
               
    return df 