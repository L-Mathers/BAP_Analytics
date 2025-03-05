# feature_extraction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_soc(
    df,
    voltage_col="voltage",
    step_col="Step Type",
    capacity_col="capacity",
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
        capacity_col (str, optional): The column name for capacity. Defaults to 'charge_capacity'.
        voltage_tolerance (float, optional): The tolerance level for voltage comparison. Defaults to 0.005.

    Returns a DataFrame with two new columns:
        'Accumulated Capacity' and 'Estimated SoC'.
    """
    # 1. Create a capacity column by integrating the time and current columns
    df['time_diff'] = df['time'].diff().fillna(0)  # Calculate time differences
    df['capacity_manual'] = (df['current'] * df['time_diff']).cumsum() / 3600  # Integrate current over time to get capacity in Ah
    
    # 2. Compute capacity difference. We take abs() then multiply by sign below.
    df["cap_diff"] = df['capacity_manual'].diff().abs().fillna(0)

    

    

    # 3. Identify boundary voltages
    min_voltage = df[voltage_col].min()
    max_voltage = df[voltage_col].max()
    print(f'Min Voltage: {min_voltage}')
    print(f'Max Voltage: {max_voltage}')
    near_min = min_voltage + voltage_tolerance
    near_max = max_voltage - voltage_tolerance

    # 4. Determine sign of each row: +1 if charge, -1 if discharge, 0 if rest/other
    step_type = df[step_col].astype(str).str.lower().fillna("")
    sign = np.where(
        step_type.str.contains("discharge"),
        -1,
        np.where(step_type.str.contains("charge"), 1, 0),
    )
    df["Sign"] = sign

    # 5. Find "last charge" and "last discharge" events:
    #       last charge -> next step is rest, and voltage >= near_max
    #       last discharge -> next step is rest, and voltage <= near_min
    #    We'll treat these as "recalibration points."
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

    # 7. Combine the two event types into a single list, labeled, and sort by index
    events = []
    for idx in charge_indices:
        events.append((idx, "charge"))
    for idx in discharge_indices:
        events.append((idx, "discharge"))
    events.sort(key=lambda x: x[0])  # sort by the row index ascending

    # If no events, do a single pass cumsum with a single nominal.
    
    if len(events) == 0:
        print("Warning: No full-charge or full-discharge event detected.")
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
        df["Estimated SoC"] = estimated_soc
        return df

    # 8. We'll keep track of a "current" nominal capacity that can be updated
    #    at each full-charge event.
    #    Start with a nominal capacity guess (e.g., maximum in the entire dataset).
    if update_nominal:
        nominal_capacity = nom_cap
    else:
        nominal_capacity = df[capacity_col].max()

    # Signed capacity changes for all rows
    raw_cap_diff = sign * df["cap_diff"].values

    # We'll accumulate segment-by-segment.
    start_idx = 0  # Start of the current segment
    current_baseline = 0.0  # Accumulated capacity at the last event

    # 8a. First partial segment: from row 0 up to (but not including) the first event
    first_event_idx = events[0][0]
    if first_event_idx > 0:
        # cumsum in [0, first_event_idx)
        segment_diffs = raw_cap_diff[start_idx:first_event_idx]
        accumulated_capacity[start_idx:first_event_idx] = current_baseline + np.cumsum(
            segment_diffs
        )
        estimated_soc[start_idx:first_event_idx] = (
            accumulated_capacity[start_idx:first_event_idx] / nominal_capacity
        ) * 100
        # Clip
        accumulated_capacity[start_idx:first_event_idx] = np.clip(
            accumulated_capacity[start_idx:first_event_idx], 0, None
        )
        estimated_soc[start_idx:first_event_idx] = np.clip(
            estimated_soc[start_idx:first_event_idx], 0, 100
        )

    # 9. Process each event in chronological order
    for i, (evt_idx, evt_type) in enumerate(events):
        # Fill cumsum from the previous event to this event index (exclusive)
        if i > 0:
            prev_event_idx = events[i - 1][0]
            segment_diffs = raw_cap_diff[prev_event_idx:evt_idx]
            

            accumulated_capacity[prev_event_idx:evt_idx] = current_baseline + np.cumsum(segment_diffs)


            estimated_soc[prev_event_idx:evt_idx] = (
                accumulated_capacity[prev_event_idx:evt_idx] / nominal_capacity
            ) * 100
            # Clip
            accumulated_capacity[prev_event_idx:evt_idx] = np.clip(
                accumulated_capacity[prev_event_idx:evt_idx], 0, None
            )
    
            estimated_soc[prev_event_idx:evt_idx] = np.clip(
                estimated_soc[prev_event_idx:evt_idx], 0, 100
            )

        # Now handle the actual event row "evt_idx"
        if evt_type == "charge":
           
            if update_nominal:
                new_nominal = df.loc[evt_idx:, capacity_col].max()
                nominal_capacity = new_nominal

                # Full charge: set capacity to max
                full_val = df.loc[evt_idx:, capacity_col].max()
                accumulated_capacity[evt_idx] = full_val

            estimated_soc[evt_idx] = 100.0
            current_baseline = accumulated_capacity[evt_idx-1]  # This should hold the max value


        elif evt_type == "discharge":
       
            accumulated_capacity[evt_idx] = 0.0
            estimated_soc[evt_idx] = 0.0
            current_baseline = 0.0  # Reset baseline for the next cycle

    


        # Move the segment start index to this event
        start_idx = evt_idx

    # 10. After the last event, fill the cumsum to the end of the dataset
    last_event_idx = events[-1][0]
    if last_event_idx < len(df) - 1:
        segment_diffs = raw_cap_diff[last_event_idx:]
        # The baseline at the last event is "current_baseline"
        # so we do baseline + cumsum of diffs
        accumulated_capacity[last_event_idx:] = current_baseline + np.cumsum(
            segment_diffs
        )
        estimated_soc[last_event_idx:] = (
            accumulated_capacity[last_event_idx:] / nominal_capacity
        ) * 100

    # 11. Final clip
    accumulated_capacity = np.clip(accumulated_capacity, 0, None)
    estimated_soc = np.clip(estimated_soc, 0, 100)

    # 12. Store in DataFrame
    df["Accumulated Capacity"] = accumulated_capacity
    df["soc"] = estimated_soc

    # Plot Accumulated Capacity
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Accumulated Capacity (Ah)", color='tab:blue')
    ax1.plot(df["time"], df["Accumulated Capacity"], label="Accumulated Capacity", color='tab:blue')
    ax1.axhline(0, color='r', linestyle='--', label='Zero Line')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Capacity (Ah)", color='tab:green')
    ax2.plot(df["time"], df['charge_capacity'], label="Capacity", color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.title("Accumulated Capacity and Capacity vs. Time")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)


   

 
    plt.show()
    
    df.to_csv("estimated_soc.csv", index=False)



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