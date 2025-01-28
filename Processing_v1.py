import numpy as np
import pandas as pd







def psuedo_limit_extraction(df: pd.DataFrame, zero_current_tolerance: float = 0.1):
    """
    This function processes a DataFrame containing 'current' and 'voltage' columns to identify and label different
    phases ('rest', 'charge', 'discharge') based on the current values and certain criteria. It ensures that phase
    changes are only accepted if there are at least 10 consecutive data points matching the new phase; otherwise,
    the previous phase is retained. The function also calculates the average of the last voltage values for each
    'charge' and 'discharge' group, excluding the first and last groups if needed.

    Args:
        df (pandas.DataFrame): The input DataFrame containing at least 'current' and 'voltage' columns.
        zero_current_tolerance (float, optional): The tolerance level below which current values are considered zero.
            Defaults to 0.015.

    Returns:
        tuple: A tuple containing:
            - df (pandas.DataFrame): The processed DataFrame with updated 'Step Type' and 'group' columns.
            - average_last_voltage_charge (float): The average of the last voltage values in each 'charge' group.
            - average_last_voltage_discharge (float): The average of the last voltage values in each 'discharge' group.

    Raises:
        ValueError: If the required columns 'current' and 'voltage' are not present in the DataFrame.
    """

    if "current" in df.columns and "voltage" in df.columns:
        current = df["current"]

        # Apply tolerance to current values to consider small values as zero
        current_abs = current.abs()
        current_value = current.where(current_abs > zero_current_tolerance, 0)

        # Assign initial 'Step Type' based on 'current_value'
        df["Step Type"] = "rest"  # Default to 'rest'
        df.loc[current_value > 0, "Step Type"] = "charge"
        df.loc[current_value < 0, "Step Type"] = "discharge"

        # Create 'group' by detecting changes in 'Step Type'
        # Each time 'Step Type' changes, increment the group number
        df["group"] = (df["Step Type"] != df["Step Type"].shift()).cumsum()

        # Get the length of each group
        group_lengths = df.groupby("group").size()

        # Temporary variable to hold the current valid phase
        current_phase = df["Step Type"].iloc[0]
        df["Adjusted Step Type"] = current_phase  # Initialize the adjusted 'Step Type'

        # Iterate over each group to validate phase changes
        for group_number, length in group_lengths.items():
            # Get indices of the current group
            group_indices = df[df["group"] == group_number].index
            # Proposed new phase based on initial assignment
            proposed_phase = df.loc[group_indices[0], "Step Type"]

            if length >= 10:
                # Accept the new phase if group length is at least 10
                current_phase = proposed_phase
            else:
                # Reject the new phase; keep the previous valid phase
                pass  # current_phase remains unchanged

            # Assign the current valid phase to the 'Adjusted Step Type' for this group
            df.loc[group_indices, "Adjusted Step Type"] = current_phase

        # Update 'Step Type' and 'group' with adjusted values
        df["Step Type"] = df["Adjusted Step Type"]
        df.drop(columns=["Adjusted Step Type"], inplace=True)
        df["group"] = (df["Step Type"] != df["Step Type"].shift()).cumsum()

        # Filter out the first and last groups if needed
        first_group = df["group"].iloc[0]
        last_group = df["group"].iloc[-1]
        filtered_groups = df[~df["group"].isin([first_group, last_group])]

        # Compute averages for 'charge' and 'discharge' steps
        # For each group, get the last voltage value and compute the mean
        average_last_voltage_charge = (
            filtered_groups[filtered_groups["Step Type"] == "charge"]
            .groupby("group")["voltage"]
            .last()
            .mean()
        )
        average_last_voltage_discharge = (
            filtered_groups[filtered_groups["Step Type"] == "discharge"]
            .groupby("group")["voltage"]
            .last()
            .mean()
        )

        return (
            df,
            round(average_last_voltage_charge, 4),
            round(average_last_voltage_discharge, 4),
        )
    else:
        raise ValueError(
            "Error: Required columns 'current' and 'voltage' not found in the DataFrame."
        )


def add_cv_steps(
    df: pd.DataFrame,
    vhigh: float,
    vlow: float,
    voltage_tolerance: float = 1e-2,
    current_delta_threshold: float = 0.05,
    voltage_delta_threshold: float = 0.0001,
):
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
    required_columns = ["Step Type", "voltage", "current", "group"]
    if not all(col in df.columns for col in required_columns):
        # print("Error: Required columns are missing in the DataFrame.")
        return df

    # Calculate the absolute change in current for each row
    df["delta_current"] = df["current"].diff().abs()
    df["delta_voltage"] = df["voltage"].diff().abs()

    # Process each group of data separately based on the 'group' column
    for group_number, group_data in df.groupby("group"):
        step_type = group_data["Step Type"].iloc[0]  # Get the 'Step Type' of the group

        # Exclude the first 10 rows of the group data
        group_data = group_data.iloc[10:]

        if step_type == "charge":
            # Condition for identifying the start of 'cv charge' phase:

            cv_condition = (group_data["delta_current"] >= current_delta_threshold) & (
                group_data["delta_voltage"] <= voltage_delta_threshold
            )

            # Find the first index where the CV condition is met
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                # Update 'Step Type' to 'charge cv' from the CV start index onwards in the group
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, "Step Type"] = "charge cv"

        elif step_type == "discharge":
            # Condition for identifying the start of 'cv discharge' phase:

            cv_condition = group_data["delta_current"] >= current_delta_threshold

            # Find the first index where the CV condition is met
            cv_start_indices = group_data.index[cv_condition]
            if not cv_start_indices.empty:
                cv_start_index = cv_start_indices[0]
                # Update 'Step Type' to 'discharge cv' from the CV start index onwards in the group
                indices_to_update = group_data.loc[cv_start_index:].index
                df.loc[indices_to_update, "Step Type"] = "discharge cv"

    # Now, remove CV sequences that are less than 10 consecutive steps
    df["cv_flag"] = (df["Step Type"].str.contains("cv")).fillna(False).astype(int)

    # Identify sequences of CV steps
    df["cv_group"] = (df["cv_flag"] != df["cv_flag"].shift()).cumsum()

    # Remove CV sequences that have less than 10 consecutive steps
    for cv_group, group_data in df.groupby("cv_group"):
        if group_data["cv_flag"].iloc[0] == 1 and len(group_data) < 10:
            # Set Step Type back to original (charge or discharge) for short CV sequences
            original_step_type = (
                df.loc[group_data.index[0] - 1, "Step Type"]
                if group_data.index[0] > 0
                else (
                    "charge"
                    if "charge cv" in group_data["Step Type"].iloc[0]
                    else "discharge"
                )
            )
            df.loc[group_data.index, "Step Type"] = original_step_type

    # Remove the temporary columns
    df.drop(columns=["delta_current", "cv_flag", "cv_group"], inplace=True)

    return df


def assign_cycle_keys(data: list, is_zp: bool):
    """
    This function assigns cycle numbers to paired 'charge' and 'discharge' entries in the given data.
    A cycle number is assigned when a 'charge' entry is followed by a 'discharge' entry. The pairing
    of 'charge' and 'discharge' is based on the sequence of data and optionally on whether the discharge
    is part of a full cycle (when `is_zp` is False).

    Args:
        data (list): A list of dictionaries, where each dictionary represents a data entry with keys such as 'group_type' and optionally 'full_cycle'.
        is_zp (bool): A boolean flag. If True, only full cycles (determined by 'full_cycle' key) are considered for pairing charges with discharges.

    Returns:
        tuple:
            - data (list): The modified data with assigned cycle numbers.
            - cycle_number (int): The next cycle number after all pairings are complete.
    """

    last_charge = None  # To track the most recent 'charge' entry
    cycle_number = 0  # Initialize cycle number

    # Iterate through each entry in the data
    for entry in data:
        # Check if the entry is a 'charge'
        if entry["group_type"] == "charge":
            last_charge = entry  # Save the 'charge' entry
        # If the entry is a 'discharge' and there is a 'charge' to pair it with
        elif entry["group_type"] == "discharge" and last_charge:
            # If not in 'zp' mode or the 'full_cycle' key is True, assign the cycle number
            if not is_zp or entry.get("full_cycle", False):
                cycle_number += 1  # Increment cycle number for the next cycle
                # Assign the same cycle number to both the last 'charge' and current 'discharge'
                last_charge["cycle"] = cycle_number
                entry["cycle"] = cycle_number
                last_charge = None  # Reset last_charge after pairing to avoid reuse

    return data, cycle_number  # Return the modified data and the final cycle number


import pandas as pd

def find_parameters_for_section(
    groups,
    targets,
    c_rate_tolerance=.2,
    raw_data=None,
    section_name=None
):
    """
    Merged logic:
      - Original per-cycle approach (to ensure correctness).
      - New time-series approach.

    Returns a DataFrame with:
      - Per-cycle rows,
      - Time-series rows,
      - Single-value rows 
              """

    # 1) Separate targets
    single_value_targets = []
    per_cycle_targets = []
    time_series_targets = []

    # Key mapping from old code
    key_mapping = {
        'c_rate': 'c-rate',
        'CCCV': 'CCCV',
        'cycle': 'cycle',
        'group_type': 'group_type',
        'soc': 'soc',
        'pulse': 'pulse',
        'interest_variable': 'interest_variable'
    }

    for t in targets:
        if t.get('per_cycle', False):
            per_cycle_targets.append(t)
        elif t.get('time_series', False):
            time_series_targets.append(t)
        else:
            single_value_targets.append(t)

    # 2) Process SINGLE-VALUE Targets 
    single_value_results = {}
    
    def process_single_value_targets():
        for targetconfig in single_value_targets:
            target = dict(targetconfig)
            target_key = target.get('key')
            interest_variable = target.get('interest_variable')

            # Criteria for matching
            ignore_keys = {'key', 'interest_variable', 'per_cycle', 'aggregation', 'time_series'}
            criteria_keys = set(target.keys()) - ignore_keys

            for group in groups:
                match = True
                for criterion in criteria_keys:
                    target_value = target[criterion]
                    group_key = key_mapping.get(criterion, criterion)
                    group_value = group.get(group_key)

                    if target_value is None:
                        # None => skip
                        continue

                    if criterion == 'c_rate':
                        if group_value is None or abs(group_value - target_value) > c_rate_tolerance:
                            match = False
                            break
                    else:
                        if group_value != target_value:
                            match = False
                            break

                # If a matching group is found, extract the interest variable
                if match:
                    if interest_variable in group:
                        single_value_results[target_key] = group[interest_variable]
                        print(f"Found {interest_variable} in group {group.get('group_number', 'N/A')}, value: {group[interest_variable]}")
                    else:
                        print(f"Warning: {interest_variable} not found in group {group.get('group_number', 'N/A')}")
                    break  # Stop after finding the first matching group

    process_single_value_targets()

    # 3) Process PER-CYCLE Targets 
    per_cycle_results = []
    
    def process_per_cycle_targets():
        # Find the set of cycles in the groups
        cycles = sorted(set(g.get('cycle') for g in groups if 'cycle' in g))

        for cycle in cycles:
            cycle_results = {}

            # Collect all groups for this cycle
            cycle_groups = [g for g in groups if g.get('cycle') == cycle]

            # For each per-cycle target
            for targetconfig in per_cycle_targets:
                target = dict(targetconfig)
                target_key = target.get('key')
                interest_variable = target.get('interest_variable')
                aggregation_method = target.get('aggregation', None)

                # Criteria for matching
                ignore_keys = {'key', 'interest_variable', 'per_cycle', 'aggregation', 'time_series'}
                criteria_keys = set(target.keys()) - ignore_keys
                matching_values = []

                for grp in cycle_groups:
                    match = True
                    for criterion in criteria_keys:
                        target_value = target[criterion]
                        group_key = key_mapping.get(criterion, criterion)
                        group_value = grp.get(group_key)

                        if target_value is None:
                            # None => skip
                            continue

                        if criterion == 'c_rate':
                            if group_value is None or abs(group_value - target_value) > c_rate_tolerance:
                                match = False
                                break
                        else:
                            if group_value != target_value:
                                match = False
                                break

                    if match:
                        if interest_variable in grp:
                            matching_values.append(grp[interest_variable])

                # Now aggregate
                if matching_values:
                    if aggregation_method == 'min':
                        cycle_results[target_key] = min(matching_values)
                    elif aggregation_method == 'max':
                        cycle_results[target_key] = max(matching_values)
                    elif aggregation_method == 'sum':
                        cycle_results[target_key] = sum(matching_values)
                    elif aggregation_method == 'average':
                        cycle_results[target_key] = sum(matching_values) / len(matching_values)
                    else:
                        # default: first
                        cycle_results[target_key] = matching_values[0]

            # Add single-value results to cycle_results
            for k, v in single_value_results.items():
                cycle_results[k] = v

            per_cycle_results.append(cycle_results)

    if per_cycle_targets:
        process_per_cycle_targets()

    # Build df_per_cycle if we have results
    if per_cycle_results:
        df_per_cycle = pd.DataFrame(per_cycle_results)
    else:
        df_per_cycle = pd.DataFrame()

    # 4) Process TIME-SERIES Targets
    df_time_series = pd.DataFrame()
    
    if time_series_targets:
        found_start_end = False
        start_idx, end_idx = None, None

        # Find the first group that matches any time-series target
        for t in time_series_targets:
            ignore_keys = {'key', 'interest_variable', 'per_cycle', 'aggregation', 'time_series'}
            criteria_keys = set(t.keys()) - ignore_keys

            for grp in groups:
                match = True
                for criterion in criteria_keys:
                    target_val = t[criterion]
                    group_key = key_mapping.get(criterion, criterion)
                    group_val = grp.get(group_key)

                    if target_val is None:
                        continue

                    if criterion == 'c_rate':
                        if group_val is None or abs(group_val - target_val) > c_rate_tolerance:
                            match = False
                            break
                    else:
                        if group_val != target_val:
                            match = False
                            break

                if match:
                    start_idx = grp.get('start_index', None)
                    end_idx   = grp.get('end_index', None)
                    if start_idx is not None and end_idx is not None and raw_data is not None:
                        found_start_end = True
                        break
            if found_start_end:
                break

        if found_start_end:
            index_range = range(start_idx, end_idx + 1)
            df_time_series = pd.DataFrame(index=index_range)

            # For each time-series target
            for t in time_series_targets:
                target_key = t.get('key')
                interest_var = t.get('interest_variable')
                if isinstance(interest_var, str) and interest_var in raw_data.columns:
                    df_time_series[target_key] = raw_data.loc[start_idx:end_idx, interest_var].values
                else:
                    # fill with None
                    df_time_series[target_key] = None

            # Also replicate single-value results in each row
            for k, v in single_value_results.items():
                df_time_series[k] = v

    

    # 6) Decide on final DataFrame
    if not df_per_cycle.empty and df_time_series.empty:
        final_df = df_per_cycle
    elif df_per_cycle.empty and not df_time_series.empty:
        final_df = df_time_series
    elif not df_per_cycle.empty and not df_time_series.empty:
        final_df = pd.concat([df_per_cycle, df_time_series], ignore_index=True)
    else:
        if single_value_results:
            final_df = pd.DataFrame(single_value_results, index=[0])                
        else:
            final_df = None

    return final_df









def data_extractor(df, cell_limits, config, test_type):
    """
    This function processes a DataFrame containing charge, discharge, and rest cycles from a battery dataset.
    It calculates various cycle metrics, energy/capacity throughput, and assigns relevant information to each group
    of data. The function then produces cumulative results by merging the new data with existing data.

    Args:
        df (pandas.DataFrame): The DataFrame containing battery cycle data with columns like 'voltage', 'current', 'time', etc.
        cell_limits (dict): Dictionary containing the voltage limits ('vlow', 'vhigh') and nominal capacity of the cell.
        config (dict): Configuration dictionary with mapping and other settings for output parameters.
        lifetime_config (dict): Configuration dictionary specifying targets for extracting data over a battery's lifetime.
        existing_df (pandas.DataFrame): The existing DataFrame containing previously stored cycle data.
        is_zp (bool): Flag indicating whether to use zero-power (ZP) mode.

    Returns:
        pandas.DataFrame: A DataFrame containing cumulative values from the new and existing data, including
                          energy throughput, capacity throughput, and other calculated metrics.
    """

    # Extract the voltage limits and nominal capacity from cell_limits
    
    nominal_capacity = cell_limits["capacity"]
    first_dch_flag = False
    # Initialize variables for tracking energy and capacity throughput
    ch_energy_throughput = 0
    dch_energy_throughput = 0
    ch_capacity_throughput = 0
    dch_capacity_throughput = 0

    # Create groups and initialize the group column
    df["group"] = np.nan

    # Extract pseudo limits and update the DataFrame accordingly
    df, psuedo_low, psuedo_high = psuedo_limit_extraction(df)

    # Add constant voltage (CV) steps depending on whether it's ZP mode or not
    if is_zp:
        df = add_cv_steps(df, vhigh, vlow)
    else:
        df = add_cv_steps(df, psuedo_high, psuedo_low)
    # Ensure the original index is preserved for future reference
    df["original_index"] = df.index

    # Group the DataFrame by the 'group' column
    grouped_df = df.groupby("group")
    group_data = []

    # Precompute voltage thresholds for defining a full cycle
    high_voltage_threshold = vhigh * 0.95  # 5% below the high limit
    low_voltage_threshold = vlow * 1.05  # 5% above the low limit

    # Initialize state of charge (SOC) and pulse counter for pulse tracking
    initial_soc = 80
    pulse_counter = 0

    # Iterate over each group in the grouped DataFrame
    for group_number, group_df in grouped_df:

        # Capture the start and end indices for the group
        start_index = group_df["original_index"].iloc[0]
        end_index = group_df["original_index"].iloc[-1]
        duration = round(
            group_df["time"].iloc[-1] - group_df["time"].iloc[0], -1
        )  # Rounded duration
        initial_capacity = [
            group_df["charge_capacity"].iloc[0],
            group_df["discharge_capacity"].iloc[0],
        ]
        initial_energy = [
            group_df["charge_energy"].iloc[0],
            group_df["discharge_energy"].iloc[0],
        ]

        # Initialize a dictionary to store group-level data
        group_dict = {
            "group_number": group_number,
            "duration": duration,
            "start_index": start_index,
            "end_index": end_index,
            "start_voltage": group_df["voltage"].iloc[0],
            "end_voltage": group_df["voltage"].iloc[-1],
            "capacity": None,
            "energy": None,
            "pulse": duration < 40,
            "full_cycle": False,
            "cc_capacity": None,
            "cv_capacity": None,
            "soc": None,  # Initialize SOC
            "cc_energy": None,
            "cv_energy": None,
            "max_temp": group_df["temperature"].max(),
            "u_min": group_df["voltage"].min(),
            "u_max": group_df["voltage"].max(),
            "i_max": group_df["current"].max(),
            "ave_cc_u": None,
            "relative_capacity": None,
            "relative_energy": None,
            "coulombic_efficiency": None,
            "energy_efficiency": None,
            "c-rate": None,
            "group_type": None,
            "ch_energy_throughput": None,
            "ch_capacity_throughput": None,
            "dch_energy_throughput": None,
            "dch_capacity_throughput": None,
            "total_capacity_throughput": None,
            "total_energy_throughput": None,
        }

        # Determine whether the group contains constant current-constant voltage (CCCV) steps
        unique_steps = group_df["Step Type"].unique()
        CCCV = any(step in unique_steps for step in ["charge cv", "discharge cv"])
        group_dict["CCCV"] = CCCV

        # Handle CCCV phases and calculate relevant metrics
        if CCCV:
            cc_phase_df = group_df[group_df["Step Type"].isin(["charge", "discharge"])]
            group_dict["c-rate"] = round(
                abs(cc_phase_df["current"].mean()) / nominal_capacity, 2
            )

            cccv_indices = group_df[
                group_df["Step Type"].isin(["charge cv", "discharge cv"])
            ].index

            if len(cccv_indices) > 0:
                # Calculate cc_capacity and cv_capacity based on the start and end of CV phase
                cv_start_idx = cccv_indices[0]
                cv_end_idx = cccv_indices[-1]

                group_dict["cv_start"] = cv_start_idx
                group_dict["cv_end"] = cv_end_idx

                group_dict["ave_cc_u"] = group_df.loc[
                    group_dict["start_index"] : cv_end_idx, "voltage"
                ].mean()
                if "charge cv" in unique_steps:
                    # Handle charge phase CCCV
                    group_dict["capacity"] = abs(
                        group_df["charge_capacity"].iloc[-1] - initial_capacity[0]
                    )
                    group_dict["energy"] = abs(
                        group_df["charge_energy"].iloc[-1] - initial_energy[0]
                    )
                    cc_capacity = abs(
                        group_df.loc[cv_start_idx, "charge_capacity"]
                        - initial_capacity[0]
                    )
                    cc_energy = abs(
                        group_df.loc[cv_start_idx, "charge_energy"] - initial_energy[0]
                    )
                    cv_energy = abs(
                        group_df.loc[cv_end_idx, "charge_energy"]
                        - group_df.loc[cv_start_idx, "charge_energy"]
                    )
                    cv_capacity = abs(
                        group_df.loc[cv_end_idx, "charge_capacity"]
                        - group_df.loc[cv_start_idx, "charge_capacity"]
                    )

                else:
                    # Handle discharge phase CCCV
                    group_dict["capacity"] = abs(
                        group_df["discharge_capacity"].iloc[-1] - initial_capacity[0]
                    )
                    group_dict["energy"] = abs(
                        group_df["discharge_energy"].iloc[-1] - initial_energy[0]
                    )
                    cc_capacity = abs(
                        group_df.loc[cv_start_idx, "discharge_capacity"]
                        - initial_capacity[1]
                    )
                    cc_energy = abs(
                        group_df.loc[cv_start_idx, "discharge_energy"]
                        - initial_energy[1]
                    )
                    cv_energy = abs(
                        group_df.loc[cv_end_idx, "discharge_energy"]
                        - group_df.loc[cv_start_idx, "discharge_energy"]
                    )
                    cv_capacity = abs(
                        group_df.loc[cv_end_idx, "discharge_capacity"]
                        - group_df.loc[cv_start_idx, "discharge_capacity"]
                    )

                group_dict["cc_capacity"] = cc_capacity
                group_dict["cv_capacity"] = cv_capacity
                group_dict["cc_energy"] = cc_energy
                group_dict["cv_energy"] = cv_energy
        else:
            group_dict["c-rate"] = round(
                abs(group_df["current"].mean()) / nominal_capacity, 2
            )
            group_dict["ave_cc_u"] = group_df["voltage"].mean()

        # Identify charge, discharge, and rest phases, updating relevant metrics
        is_charge_phase = group_df["Step Type"].str.contains(
            r"\bcharge\b", case=False, regex=True
        )
        is_discharge_phase = group_df["Step Type"].str.contains(
            r"\bdischarge\b", case=False, regex=True
        )

        if is_charge_phase.any():
            # Check if it's a full charge cycle
            start_voltage = (
                group_data[-1]["start_voltage"]
                if group_data
                else group_dict["start_voltage"]
            )
            is_full_cycle = (
                start_voltage <= low_voltage_threshold
                and group_dict["end_voltage"] >= high_voltage_threshold
            )
            group_dict["full_cycle"] = is_full_cycle
            group_dict["group_type"] = "charge"

            # Update charge-related metrics
            if group_dict["cc_capacity"] is None:
                group_dict["capacity"] = abs(
                    group_df["discharge_capacity"].iloc[-1] - initial_capacity[0]
                )
                group_dict["energy"] = abs(
                    group_df["discharge_energy"].iloc[-1] - initial_energy[0]
                )
                group_dict["cc_energy"] = group_dict["energy"]
                group_dict["cc_capacity"] = group_dict["capacity"]

            ch_energy_throughput += group_dict["energy"]
            ch_capacity_throughput += group_dict["capacity"]

            group_dict['ch_energy_throughput'] = ch_energy_throughput/ 1000
            group_dict['ch_capacity_throughput'] = ch_capacity_throughput /1000
           

        if is_discharge_phase.any():

            # Check if it's a full discharge cycle
            start_voltage = (
                group_data[-1]["start_voltage"]
                if group_data
                else group_dict["start_voltage"]
            )
            is_full_cycle = (
                start_voltage >= high_voltage_threshold
                and group_dict["end_voltage"] <= low_voltage_threshold
            )
            group_dict["full_cycle"] = is_full_cycle
            group_dict["group_type"] = "discharge"

            # Update discharge-related metrics
            if group_dict["cc_capacity"] is None:
                group_dict["capacity"] = abs(
                    group_df["charge_capacity"].iloc[-1] - initial_capacity[1]
                )
                group_dict["energy"] = abs(
                    group_df["charge_energy"].iloc[-1] - initial_energy[1]
                )
                group_dict["cc_energy"] = group_dict["energy"]
                group_dict["cc_capacity"] = group_dict["capacity"]

            dch_energy_throughput += group_dict["energy"]
            dch_capacity_throughput += group_dict["capacity"]

            group_dict['dch_energy_throughput'] = dch_energy_throughput/1000
            group_dict['dch_capacity_throughput'] = dch_capacity_throughput/1000

            if first_dch_flag == False:
                first_cycle_cap = group_dict["capacity"]
                first_cycle_en = group_dict["energy"]
                group_dict["relative_capacity"] = 100
                group_dict["relative_energy"] = 100
                first_dch_flag = True
            else:
                group_dict["relative_capacity"] = (
                    group_dict["capacity"] / first_cycle_cap
                ) * 100
                group_dict["relative_energy"] = (
                    group_dict["energy"] / first_cycle_en
                ) * 100

        if not is_charge_phase.any() and not is_discharge_phase.any():
            group_dict["group_type"] = "rest"

        # Handle SOC value for pulses and update pulse-related metrics
        if group_dict["pulse"]:
            group_dict["soc"] = initial_soc - (pulse_counter * 30)
            pulse_counter += 1

            # Calculate internal resistance based on voltage drop during pulses
            v1 = group_df["voltage"].iloc[0]
            pulse_length = [10, 2, 0.1]
            for i in pulse_length:
                seconds_later = group_df[
                    group_df["time"] >= group_df["time"].iloc[0] + i
                ]
                if not seconds_later.empty:
                    v2 = seconds_later["voltage"].iloc[0]
                    I = group_df["current"].mean()
                    group_dict[f"internal_resistance_{i}s"] = (
                        (v1 - v2) / abs(I) * 1000
                    )  # covert to mOhm

        group_data.append(group_dict)

    # Assign cycle numbers to the data and calculate total duration
    group_data, max_cycles = assign_cycle_keys(group_data, is_zp)
    total_duration = (
        df.loc[group_data[-1]["end_index"], "time"] / 86400
    )  # Convert time to days

    # Calculate total energy throughput and total capacity (Ah) throughput
    total_energy_throughput = ch_energy_throughput + dch_energy_throughput
    total_capacity_throughput = ch_capacity_throughput + dch_capacity_throughput
    eq_cycle = dch_capacity_throughput / nominal_capacity

    # Create a summary group with cumulative values
    summary_group = {
        "group_number": len(group_data) + 1,
        "group_type": "summary",
        "max_cycles": max_cycles,
        "total_duration": total_duration,
        "ch_energy_throughput": ch_energy_throughput / 1000,  # Convert to kWh
        "dch_energy_throughput": dch_energy_throughput / 1000,  # Convert to kWh
        "total_energy_throughput": total_energy_throughput / 1000,  # Convert to kWh
        "ch_capacity_throughput": ch_capacity_throughput / 1000,  # Convert to kAh
        "dch_capacity_throughput": dch_capacity_throughput / 1000,  # Convert to kAh
        "total_capacity_throughput": total_capacity_throughput / 1000,  # Convert to kAh
        "eq_cycle": eq_cycle,
    }
    group_data.append(summary_group)

    # Calculate parameters that depend on two different cycles (e.g., coulombic efficiency)
    if not is_zp:
        # Create a mapping from cycles to groups
        cycle_groups = {}
        for group in group_data: 
            cycle_number = group.get("cycle")
            if cycle_number is not None:
                if cycle_number not in cycle_groups:
                    cycle_groups[cycle_number] = []
                cycle_groups[cycle_number].append(group)

        # Iterate over cycles to calculate parameters
        for cycle_number, groups in cycle_groups.items():
            charge_group = next(
                (g for g in groups if g["group_type"] == "charge"), None
            )
            discharge_group = next(
                (g for g in groups if g["group_type"] == "discharge"), None
            )

            if charge_group and discharge_group:
                # Coulombic efficiency
                ch_total_capacity = charge_group.get("capacity")
                dch_total_capacity = discharge_group.get("capacity")
                if ch_total_capacity and dch_total_capacity:
                    coulombic_efficiency = (
                        dch_total_capacity / ch_total_capacity
                    ) * 100
                    charge_group["coulombic_efficiency"] = coulombic_efficiency
                    discharge_group["coulombic_efficiency"] = coulombic_efficiency

                # Energy efficiency
                ch_total_energy = charge_group.get("energy")
                dch_total_energy = discharge_group.get("energy")
                if ch_total_energy and dch_total_energy:
                    energy_efficiency = (dch_total_energy / ch_total_energy) * 100
                    charge_group["energy_efficiency"] = energy_efficiency
                    discharge_group["energy_efficiency"] = energy_efficiency

                   # Energy throughput
                ch_energy_throughput = charge_group.get('ch_energy_throughput')
                dch_energy_throughput = discharge_group.get('dch_energy_throughput')
                if ch_energy_throughput and dch_energy_throughput:
                    total_energy_throughput = ch_energy_throughput + dch_energy_throughput
                    charge_group['total_energy_throughput'] = total_energy_throughput
                    discharge_group['total_energy_throughput'] = total_energy_throughput

                # Capacity throughput
                ch_capacity_throughput = charge_group.get('ch_capacity_throughput')
                dch_capacity_throughput = discharge_group.get('dch_capacity_throughput')
                if ch_capacity_throughput and dch_capacity_throughput:
                    total_capacity_throughput = ch_capacity_throughput + dch_capacity_throughput
                    charge_group['total_capacity_throughput'] = total_capacity_throughput
                    discharge_group['total_capacity_throughput'] = total_capacity_throughput

    # Extract initial parameters and compute cumulative values

    section_df = find_parameters_for_section(
        group_data,
        config["targets"],
        raw_data=df,
        section_name="lifetime",
    )

    

    # Return the updated DataFrames
    return section_df

from fuzzywuzzy import process

def process_lifetime_test(
    data: pd.DataFrame,
    cell_limits: dict,
    config: dict,
    test_type: str,
):
    """
    Processes a lifetime battery test, extracts relevant data such as current, voltage,
    capacity, and energy, and ensures robustness against slight variations in column names
    using fuzzy matching.

    Args:
        data (pd.DataFrame): Dataframe containing the test data.
        cell_limits (dict): A dictionary containing the limits for the cell (e.g., voltage limits, nominal capacity).
        config (dict): Configuration dictionary with mapping and other settings for output parameters.
        test_type (str): Identifier for the test type.

    Returns:
        pandas.DataFrame: A DataFrame containing cumulative values calculated from the test data.
    """

    # Define required columns and their canonical names
    required_columns = {
        "time": ["test time", "time (s)", "elapsed time"],
        "current": ["current (a)", "current"],
        "voltage": ["voltage (v)", "voltage"],
        "discharge_capacity": ["discharge capacity (ah)", "capacity (ah)", "dcap"],
        "charge_capacity": ["charge capacity (ah)", "capacity (ah)", "ccap"],
        "discharge_energy": ["discharge energy (wh)", "energy (wh)", "denergy"],
        "charge_energy": ["charge energy (wh)", "energy (wh)", "cenergy"],
    }

    # Fuzzy match columns in the input data to the required columns
    matched_columns = {}
    for canonical_name, possible_names in required_columns.items():
        # Find the best match from the possible names for each column
        match = process.extractOne(
            canonical_name, data.columns, scorer=process.token_sort_ratio
        )
        if match and match[1] >= 80:  # Set a confidence threshold
            matched_columns[canonical_name] = match[0]
        else:
            raise ValueError(f"Missing or unmatched column for: {canonical_name}")

    # Rename the matched columns in the DataFrame
    data = data.rename(columns={v: k for k, v in matched_columns.items()})

    # Identify columns containing temperature data (assumed to have 'temp' in their name)
    degc_columns = [col for col in data if "temp" in col.lower()]

    # Check if any temperature columns exist and calculate the maximum temperature
    max_values = {col: max(data[col]) for col in degc_columns if col in data}

    # Find the column with the highest maximum value and assign it as the temperature column in df
    if max_values:
        max_column = max(max_values, key=max_values.get)
        data["temperature"] = data[max_column]

    # Pass the processed data through the data_extractor function for further processing and return the results
    results = data_extractor(
        data, cell_limits, config, test_type
    )

    return results


