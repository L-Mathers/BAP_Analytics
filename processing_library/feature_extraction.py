# feature_extraction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    df['time_diff'] = df['time'].diff().fillna(0)  # Calculate time differences
    df['capacity_manual'] = (df['current'] * df['time_diff']).cumsum() / 3600  # Integrate current over time to get capacity in Ah
    
    # Create a unified capacity difference column based on charge and discharge capacity
    if has_charge_cap and has_discharge_cap:
        # Calculate differences in charge and discharge capacity
        df['charge_cap_diff'] = df[charge_capacity_col].diff().fillna(0)
        df['discharge_cap_diff'] = df[discharge_capacity_col].diff().fillna(0)
        
        # Handle capacity resets by detecting when absolute capacity decreases
        # For charge capacity
        charge_reset_mask = df[charge_capacity_col].abs() < df[charge_capacity_col].shift(1).abs()
        df.loc[charge_reset_mask, 'charge_cap_diff'] = df.loc[charge_reset_mask, charge_capacity_col].abs()
        
        # For discharge capacity
        discharge_reset_mask = df[discharge_capacity_col].abs() < df[discharge_capacity_col].shift(1).abs()
        df.loc[discharge_reset_mask, 'discharge_cap_diff'] = df.loc[discharge_reset_mask, discharge_capacity_col].abs()
        
        # Create a unified capacity difference based on step type
        df['unified_cap_diff'] = 0.0
        
        # For charge steps, use charge capacity difference
        charge_mask = df[step_col].str.contains('charge', case=False, na=False)
        df.loc[charge_mask, 'unified_cap_diff'] = df.loc[charge_mask, 'charge_cap_diff'].abs()
        
        # For discharge steps, use discharge capacity difference
        discharge_mask = df[step_col].str.contains('discharge', case=False, na=False)
        df.loc[discharge_mask, 'unified_cap_diff'] = df.loc[discharge_mask, 'discharge_cap_diff'].abs()
        
        # Use the unified capacity difference for cap_diff
        df["cap_diff"] = df['unified_cap_diff']
    else:
        # Use the capacity column if charge/discharge capacity columns aren't available
        # First check if capacity column exists
        if capacity_col in df.columns:
            # Calculate capacity difference
            df['capacity_diff'] = df[capacity_col].diff().fillna(0)
            
            # Handle capacity resets by detecting when absolute capacity decreases
            capacity_reset_mask = df[capacity_col].abs() < df[capacity_col].shift(1).abs()
            df.loc[capacity_reset_mask, 'capacity_diff'] = df.loc[capacity_reset_mask, capacity_col].abs()
            
            # Use the capacity difference for cap_diff
            df["cap_diff"] = df['capacity_diff'].abs()
        else:
            # Fallback to manual calculation from current integration
            df["cap_diff"] = df['capacity_manual'].diff().abs().fillna(0)

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
        (sign == 1)
        & (next_step_type.str.contains("rest"))
        & (df[voltage_col] >= near_max)
    )
    is_last_discharge_step = (
        (sign == -1)
        & (next_step_type.str.contains("rest"))
        & (df[voltage_col] <= near_min)
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
        print(f"Using nominal capacity: {nominal_capacity}")
    else:
        nominal_capacity = df[capacity_col].max()
        print(f"Using maximum capacity: {nominal_capacity}")

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
            print(f"Shifting first segment [0:{first_event_idx}] up by {shift_amount} to avoid negative values")
            segment_accumulated += shift_amount
        
        # Assign the shifted values back to accumulated_capacity
        accumulated_capacity[start_idx:first_event_idx] = segment_accumulated
        print(f"Set accumulated capacity for first segment [0:{first_event_idx}], min value: {np.min(segment_accumulated)}, values: {segment_accumulated[::100]}")
        
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
                print(f"Shifting segment [{prev_event_idx}:{evt_idx}] up by {shift_amount} to avoid negative values")
                segment_accumulated += shift_amount
            
            # Assign the shifted values back to accumulated_capacity
            accumulated_capacity[prev_event_idx:evt_idx] = segment_accumulated
            print(f"Set accumulated capacity for segment [{prev_event_idx}:{evt_idx}], baseline: {current_baseline}, min value: {np.min(segment_accumulated)}, values: {segment_accumulated[::100]}")
            
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
            
            nominal_capacity = accumulated_capacity[evt_idx-1]
            current_baseline = accumulated_capacity[evt_idx-1]
            print(f"Current baseline: {current_baseline}")

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
            print(f"Shifting final segment [{last_event_idx}:end] up by {shift_amount} to avoid negative values")
            segment_accumulated += shift_amount
        
        # Assign the shifted values back to accumulated_capacity
        accumulated_capacity[last_event_idx:] = segment_accumulated
        print(f"Set accumulated capacity for final segment [{last_event_idx}:end], min value: {np.min(segment_accumulated)}, values: {segment_accumulated[::100]}")
        
        estimated_soc[last_event_idx:] = (
            accumulated_capacity[last_event_idx:] / nominal_capacity
        ) * 100

    # Final check for any remaining negative values (should not happen with our shifting approach)
    min_accumulated = np.min(accumulated_capacity)
    if min_accumulated < 0:
        print(f"Warning: Negative values still found in accumulated capacity: {min_accumulated}")
        accumulated_capacity = np.clip(accumulated_capacity, 0, None)
        print(f"Applied final clipping to ensure no negative values")
    
    print(f"Final accumulated capacity range: [{np.min(accumulated_capacity)}, {np.max(accumulated_capacity)}]")
    
    # Ensure SOC is within 0-100%
    estimated_soc = np.clip(estimated_soc, 0, 100)
    print(f"Final SOC range: [{np.min(estimated_soc)}, {np.max(estimated_soc)}]")

    # Store in DataFrame
    df["Accumulated Capacity"] = accumulated_capacity
    df["soc"] = estimated_soc
    df.to_csv('soc-analysis-V2.csv', index=False)
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    # Create subplot layout (2 rows, 1 column)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Battery Capacity", "State of Charge (SOC)"))

    # Plot Capacity and Accumulated Capacity
    fig.add_trace(go.Scatter(
        x=df.index, y=df[capacity_col], mode='lines', name='Capacity', opacity=0.7),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Accumulated Capacity'], mode='lines', name='Accumulated Capacity', opacity=0.7),
        row=1, col=1)

    # Plot SOC
    fig.add_trace(go.Scatter(
        x=df.index, y=df['soc'], mode='lines', name='SOC', line=dict(color='green'), opacity=0.7),
        row=2, col=1)

    # Update layout
    fig.update_layout(
        height=700,
        width=1000,
        title_text='Battery Capacity and State of Charge Analysis',
        showlegend=True
    )

    fig.update_yaxes(title_text='Capacity (Ah)', row=1, col=1)
    fig.update_yaxes(title_text='State of Charge (%)', row=2, col=1)
    fig.update_xaxes(title_text='Index', row=2, col=1)

    # Save as HTML
    fig.write_html("cycle2_battery_analysis_plot.html")

    # Optional: display in Jupyter Notebook
    # fig.show()
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
            elif is_rpt:
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