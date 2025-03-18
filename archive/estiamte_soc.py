def estimate_soc(
    df,
    nominal_capacity,
    voltage_col="voltage",
    step_col="Step Type",
    capacity_col="charge_capacity",
    voltage_tolerance=0.0015,
):
    """
    Estimate the State of Charge (SoC) for a battery test dataset, recalibrating
    whenever a 'last charge' (full) or 'last discharge' (empty) event is detected.

    Nominal capacity is updated at each 'last charge' event, using:
        new_nominal = df.loc[idx:, capacity_col].max()
    and that new nominal capacity is used *immediately* for subsequent rows.

    Args:
        df (pandas.DataFrame): The input DataFrame containing battery test data.
        nominal_capacity (float): The nominal capacity as obtained from the accompanying document.
        voltage_col (str, optional): The column name for voltage. Defaults to 'voltage'.
        step_col (str, optional): The column name for step type. Defaults to 'Step Type'.
        capacity_col (str, optional): The column name for capacity. Defaults to 'charge_capacity'.
        voltage_tolerance (float, optional): The tolerance level for voltage comparison. Defaults to 0.005.

    Returns a DataFrame with two new columns:
        'Accumulated Capacity' and 'Estimated SoC'.
    """

    # 2. Compute capacity difference. We take abs() then multiply by sign below.
    df["cap_diff"] = df[capacity_col].diff().abs().fillna(0)

    # 3. Identify boundary voltages
    min_voltage = df[voltage_col].min()
    max_voltage = df[voltage_col].max()
    near_min = min_voltage + voltage_tolerance
    near_max = max_voltage - voltage_tolerance

    # 4. Determine sign of each row: +1 if charge, -1 if discharge, 0 if rest/other
    step_type = df[step_col].astype(str).str.lower().fillna("")
    sign = np.where(
        step_type.str.contains("discharge"),
        -1,
        np.where(step_type.str.contains("charge"), 1, 0),
    )

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

    # 6. Prepare arrays for final results
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
        logger.warning("No full-charge or full-discharge event detected.")
        # We'll just pick the largest capacity in the entire dataset as nominal:
        # Signed capacity changes
        raw_cap_diff = sign * df["cap_diff"].values
        accumulated_capacity[:] = np.cumsum(raw_cap_diff)
        estimated_soc[:] = (accumulated_capacity / nominal_capacity) * 100
        # Clip
        accumulated_capacity = np.clip(accumulated_capacity, 0, None)
        estimated_soc = np.clip(estimated_soc, 0, 100)
        df["soc"] = estimated_soc
        return df

    # 8. We'll keep track of a "current" nominal capacity that can be updated
    #    at each full-charge event.
    #    Start with a nominal capacity guess (e.g., maximum in the entire dataset).

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
            # The previous event index
            prev_event_idx = events[i - 1][0]
            segment_diffs = raw_cap_diff[prev_event_idx:evt_idx]
            accumulated_capacity[prev_event_idx:evt_idx] = current_baseline + np.cumsum(
                segment_diffs
            )
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
        if evt_type == "charge" and (df[capacity_col][evt_idx]/nominal_capacity > 0.9):
            print('event index', evt_idx)
            print('capacity at event index', df[capacity_col][evt_idx])
            print('capacity at event index - 1', df[capacity_col][evt_idx-1])
            print('nominal capacity', nominal_capacity)
            print('percent of nominal capacity', df[capacity_col][evt_idx]/nominal_capacity)
            print('accumulated capacity at event index', accumulated_capacity[evt_idx-1])
            # Full charge: set capacity to whatever is in the capacity_col at evt_idx
            print('voltage at event index', df[voltage_col][evt_idx], 'near_max', near_max)
            print(f"Full charge event at index {evt_idx}, capacity = {df.loc[evt_idx:, capacity_col].max()}")
            
            full_val = df.loc[evt_idx:, capacity_col].max()
            accumulated_capacity[evt_idx] = full_val
            nominal_capacity = full_val

            # SoC at a full charge is 100%
            estimated_soc[evt_idx] = 100.0

            # Update baseline for the next segment
            current_baseline = accumulated_capacity[evt_idx]

        elif evt_type == "discharge":
            # Full discharge: set capacity = 0
            accumulated_capacity[evt_idx] = 0.0
            # SoC at a full discharge is 0%
            estimated_soc[evt_idx] = 0.0

            # Update baseline for the next segment
            current_baseline = 0.0

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
    df.to_csv('/Users/liammathers/Desktop/Github/bmw_lifetime_processing_int/temp_data/soc.csv')
    return df