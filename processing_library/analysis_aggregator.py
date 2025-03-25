# analysis_aggregator.py

import numpy as np
import pandas as pd


def find_parameters_for_section(groups, targets, c_rate_tolerance=0.5, raw_data=None):
    """
    Produces a final DataFrame in three vertical 'blocks':
      1) Single-value rows at the top
      2) Per-cycle rows in the middle
      3) Time-series rows at the bottom
    """

    # if "targets" not in targets:
    #     # If for some reason the config doesn't have "targets"
    #     return None

    single_value_targets = []
    per_cycle_targets = []
    time_series_targets = []

    # Separate the three categories
    for t in targets:
        if t.get("per_cycle", False):
            per_cycle_targets.append(t)

        elif t.get("time_series", False):
            time_series_targets.append(t)

        else:
            single_value_targets.append(t)

    print(f"Found {len(single_value_targets)} single-value targets")
    print(f"Found {len(per_cycle_targets)} per-cycle targets")
    print(f"Found {len(time_series_targets)} time-series targets")

    # 1) SINGLE-VALUE PARTIAL DF
    df_sv = _build_single_value_df(groups, single_value_targets, c_rate_tolerance)

    # 2) PER-CYCLE PARTIAL DF
    df_pc = _build_per_cycle_df(groups, per_cycle_targets, c_rate_tolerance)

    # 3) TIME-SERIES PARTIAL DF
    df_ts = _build_time_series_df(groups, time_series_targets, c_rate_tolerance, raw_data)

    # Concatenate them in the order: single-value, per-cycle, time-series
    partial_dfs = [df_sv, df_pc, df_ts]
    final_df = columnwise_stack_no_top_padding(partial_dfs)
    if final_df.empty:
        return None
    return final_df


def columnwise_stack_no_top_padding(dataframes):
    """
    Stacks columns top-to-bottom while preserving row relationships.
    For related values (those from the same group), they will appear in the same row.
    Missing columns => NaN
    """
    if not dataframes:
        return pd.DataFrame()

    # Identify all columns across all dataframes
    all_cols = set()
    for df in dataframes:
        all_cols.update(df.columns)
    all_cols = sorted(all_cols)

    # Start with empty data dictionary
    col_data = {col: [] for col in all_cols}

    # Process each dataframe sequentially
    for df in dataframes:
        if df.empty:
            continue

        # Get current length of our result
        current_len = max(len(vals) for vals in col_data.values()) if any(col_data.values()) else 0

        # For each row in the dataframe
        for idx, row in df.iterrows():
            # Skip rows that are all NaN
            if row.isna().all():
                continue

            # Add data from this row to each column
            for col in all_cols:
                # Ensure all columns are same length before adding new data
                if len(col_data[col]) < current_len:
                    col_data[col].extend([np.nan] * (current_len - len(col_data[col])))

                # Add the value if column exists, otherwise NaN
                if col in df.columns:
                    col_data[col].append(row[col])
                else:
                    col_data[col].append(np.nan)

    # Ensure all columns are of equal length in the final result
    max_len = max(len(vals) for vals in col_data.values()) if col_data else 0
    for col, vals in col_data.items():
        short_by = max_len - len(vals)
        if short_by > 0:
            vals.extend([np.nan] * short_by)

    # Create the DataFrame and drop any all-NaN rows
    df = pd.DataFrame(col_data)
    return df.dropna(how="all")


def _get_criteria_dict(target):
    """
    Extract criteria from a target definition, excluding metadata fields.

    Standardizes the criteria fields so that related targets (like DCIR and normalized DCIR)
    will be grouped together properly.
    """
    # Fields that define target metadata, not matching criteria
    ignore_keys = {
        "key",
        "interest_variable",
        "per_cycle",
        "aggregation",
        "time_series",
        "from_group",
    }

    # Extract the criteria but normalize them
    criteria = {}

    # Group type is a primary grouping factor
    if "group_type" in target:
        criteria["group_type"] = target["group_type"]

    # Pulse status affects which measurements are available
    if "pulse" in target:
        criteria["pulse"] = target["pulse"]

    # SOC value is critical for DCIR and normalized DCIR to match
    if "soc" in target:
        criteria["soc"] = target["soc"]

    # C-rate is important for regular measurements
    if "crate" in target:
        criteria["crate"] = target["crate"]

    # Test type is used for some specific cases
    if "test_type" in target:
        criteria["test_type"] = target["test_type"]

    # Add any other criteria fields that aren't in the ignore list
    for key, value in target.items():
        if key not in ignore_keys and key not in criteria:
            criteria[key] = value

    return criteria


def _build_single_value_df(groups, targets, c_rate_tol):
    """
    Builds a wide DataFrame where each single-value target key is its own column.
    Values from different groups are kept in separate rows, but values from the
    same group are placed in the same row.
    """
    if not targets:
        return pd.DataFrame()

    all_keys = [t["key"] for t in targets]

    # First, create a lookup of group index to group object for easier reference
    group_dict = {i: grp for i, grp in enumerate(groups)}

    # Find matching groups for each target
    target_to_groups = {}
    for i, tgt in enumerate(targets):
        matching_groups = []
        exact_matching_groups = []  # Track groups with exact SOC matches
        tgt_criteria = _get_criteria_dict(tgt)

        # First pass: try to find exact SOC matches
        for grp_idx, grp in group_dict.items():
            exact_match = True
            for c, tval in tgt_criteria.items():
                gval = grp.get(c, None)

                if c == "crate":
                    if isinstance(tval, list):
                        if len(tval) == 1:
                            if gval is None or abs(gval - tval[0]) > c_rate_tol:
                                exact_match = False
                                break
                        else:
                            lo, hi = tval
                            if gval is None or not (lo <= gval <= hi):
                                exact_match = False
                                break
                    else:
                        if gval is None or abs(gval - tval) > c_rate_tol:
                            exact_match = False
                            break
                elif c == "soc":
                    # For SOC, only exact matches in this pass
                    if gval != tval:
                        exact_match = False
                        break
                else:
                    if gval != tval:
                        exact_match = False
                        break

            if exact_match:
                exact_matching_groups.append(grp_idx)

        # If exact matches found for SOC, use those only
        if any(tgt_criteria.get("soc") is not None for _ in [0] if exact_matching_groups):
            matching_groups = exact_matching_groups
        else:
            # Second pass: if no exact SOC matches, use tolerance-based matching
            for grp_idx, grp in group_dict.items():
                match = True
                for c, tval in tgt_criteria.items():
                    gval = grp.get(c, None)

                    if c == "crate":
                        if isinstance(tval, list):
                            if len(tval) == 1:
                                if gval is None or abs(gval - tval[0]) > c_rate_tol:
                                    match = False
                                    break
                            else:
                                lo, hi = tval
                                if gval is None or not (lo <= gval <= hi):
                                    match = False
                                    break
                        else:
                            if gval is None or abs(gval - tval) > c_rate_tol:
                                match = False
                                break
                    # Special handling for SOC with tighter tolerance
                    elif c == "soc" and gval is not None and tval is not None:
                        if abs(gval - tval) > 0.5:  # Reduce to 0.5% SOC tolerance
                            match = False
                            break
                    else:
                        if gval != tval:
                            match = False
                            break

                if match:
                    matching_groups.append(grp_idx)

        target_to_groups[i] = matching_groups

    # Group targets that share the same matching groups
    groups_to_targets = {}
    for tgt_idx, group_indices in target_to_groups.items():
        group_key = tuple(sorted(group_indices))
        if group_key not in groups_to_targets:
            groups_to_targets[group_key] = []
        groups_to_targets[group_key].append(tgt_idx)

    # Build rows based on common matching groups
    rows = []
    for group_indices, target_indices in groups_to_targets.items():
        if not group_indices:  # Skip if no matching groups
            continue

        # Create a row for each matching group
        for grp_idx in group_indices:
            grp = group_dict[grp_idx]
            row = {k: np.nan for k in all_keys}

            # Fill in values for all targets that match this group
            found_values = False
            for tgt_idx in target_indices:
                tgt = targets[tgt_idx]
                tkey = tgt["key"]
                ivar = tgt.get("interest_variable")

                if ivar in grp:
                    row[tkey] = grp[ivar]
                    found_values = True

            # Only add the row if at least one value was found
            if found_values:
                rows.append(row)

    # Convert to DataFrame
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=all_keys)


def _build_per_cycle_df(groups, targets, c_rate_tol):
    """
    Builds a DataFrame with one row per cycle. Each target key is a column.
    """
    if not targets:
        return pd.DataFrame()

    all_keys = [t["key"] for t in targets]
    # unique cycles
    cycles = sorted(set(g.get("cycle") for g in groups if g.get("cycle") is not None))
    if not cycles:
        return pd.DataFrame()

    df_pc = pd.DataFrame(np.nan, index=range(len(cycles)), columns=all_keys)
    cycle_to_index = {cyc: i for i, cyc in enumerate(cycles)}

    for tgt in targets:
        tkey = tgt["key"]
        ivar = tgt.get("interest_variable")
        agg = tgt.get("aggregation", None)

        ignore_keys = {"key", "interest_variable", "per_cycle", "aggregation", "time_series"}
        crit_keys = set(tgt.keys()) - ignore_keys

        for grp in groups:
            cyc = grp.get("cycle")
            if cyc is None:
                continue
            match = True
            for c in crit_keys:
                tval = tgt[c]
                gval = grp.get(c, None)
                if c == "c_rate":
                    if gval is None or abs(gval - tval) > c_rate_tol:
                        match = False
                        break
                elif c == "range_c_rate":
                    lo, hi = tval
                    if gval is None or not (lo <= gval <= hi):
                        match = False
                        break
                else:
                    if gval != tval:
                        match = False
                        break

            if match and ivar in grp:
                val = grp[ivar]
                row_idx = cycle_to_index[cyc]
                df_pc.at[row_idx, tkey] = val

    df_pc = df_pc.drop(df_pc.index[0])
    return df_pc


def _build_time_series_df(groups, targets, c_rate_tol, raw_data):
    """
    Builds a wide DataFrame with potential time-series data for each target key.
    When 'from_group' is true, the static value from the group is attached to
    every row of time-series data from that group.
    """
    if not targets or raw_data is None:
        return pd.DataFrame()

    # Separate targets into from_group and regular targets
    from_group_targets = [t for t in targets if t.get("from_group", False)]
    regular_targets = [t for t in targets if not t.get("from_group", False)]

    # If no regular time-series targets, nothing to do
    if not regular_targets:
        return pd.DataFrame()

    all_keys = [t["key"] for t in targets]

    # First collect all time-series data rows
    time_series_rows = []

    # Process regular time-series targets to find the matching groups
    groups_with_time_series = {}  # Maps group_idx -> target_key -> time_series_data
    for tgt in regular_targets:
        tkey = tgt["key"]
        ivar = tgt.get("interest_variable")

        ignore_keys = {
            "key",
            "interest_variable",
            "per_cycle",
            "aggregation",
            "time_series",
            "from_group",
        }
        crit_keys = set(tgt.keys()) - ignore_keys

        # Find matching group
        matched_group_idx = None
        for idx, grp in enumerate(groups):
            match = True
            for c in crit_keys:
                tval = tgt[c]
                gval = grp.get(c, None)
                if c == "crate":
                    if isinstance(tval, list):
                        if len(tval) == 1:
                            if abs(gval - tval[0]) > c_rate_tol:
                                match = False
                                break
                        else:
                            lo, hi = tval
                            if gval is None or not (lo <= gval <= hi):
                                match = False
                                break
                    else:
                        if gval is None or abs(gval - tval) > c_rate_tol:
                            match = False
                            break
                else:
                    if gval != tval:
                        match = False
                        break
            if match:
                matched_group_idx = idx
                break

        if matched_group_idx is not None:
            matched_group = groups[matched_group_idx]
            s_i = matched_group["start_index"]
            e_i = matched_group["end_index"]
            if s_i is not None and e_i is not None and s_i <= e_i:
                sub_df = raw_data.loc[s_i:e_i]
                if ivar and ivar in raw_data.columns:
                    values = sub_df[ivar]
                else:
                    if "voltage" in sub_df.columns:
                        values = sub_df["voltage"]
                    else:
                        values = pd.Series([np.nan] * (e_i - s_i + 1), index=sub_df.index)

                # Store this time-series data
                if matched_group_idx not in groups_with_time_series:
                    groups_with_time_series[matched_group_idx] = {}

                groups_with_time_series[matched_group_idx][tkey] = values

    # Now collect group property values from from_group targets
    group_properties = {}  # Maps group_idx -> target_key -> static_value
    for tgt in from_group_targets:
        tkey = tgt["key"]
        ivar = tgt.get("interest_variable")

        ignore_keys = {
            "key",
            "interest_variable",
            "per_cycle",
            "aggregation",
            "time_series",
            "from_group",
        }
        crit_keys = set(tgt.keys()) - ignore_keys

        # For each group in our time-series data, check if it matches this from_group target
        for group_idx in groups_with_time_series:
            grp = groups[group_idx]
            match = True
            for c in crit_keys:
                tval = tgt[c]
                gval = grp.get(c, None)
                if c == "crate":
                    if isinstance(tval, list):
                        if len(tval) == 1:
                            if abs(gval - tval[0]) > c_rate_tol:
                                match = False
                                break
                        else:
                            lo, hi = tval
                            if gval is None or not (lo <= gval <= hi):
                                match = False
                                break
                    else:
                        if gval is None or abs(gval - tval) > c_rate_tol:
                            match = False
                            break
                else:
                    if gval != tval:
                        match = False
                        break

            if match and ivar in grp:
                if group_idx not in group_properties:
                    group_properties[group_idx] = {}

                group_properties[group_idx][tkey] = grp[ivar]

    # Build rows combining time-series data with from_group properties
    rows = []
    for group_idx, target_series in groups_with_time_series.items():
        # Determine length of time-series
        series_len = max(len(series) for series in target_series.values())

        # Get properties for this group
        properties = group_properties.get(group_idx, {})

        # Create a row for each time point
        for i in range(series_len):
            row = {k: np.nan for k in all_keys}

            # Add time-series values
            for tkey, series in target_series.items():
                if i < len(series):
                    row[tkey] = series.iloc[i]

            # Add static properties from group
            for prop_key, value in properties.items():
                row[prop_key] = value

            # Only add row if it has at least one non-NaN value
            if not all(pd.isna(val) for val in row.values()):
                rows.append(row)

    # Convert to DataFrame
    if rows:
        df = pd.DataFrame(rows)
        # Remove any remaining all-NaN rows
        return df.dropna(how="all")
    else:
        return pd.DataFrame(columns=all_keys)
