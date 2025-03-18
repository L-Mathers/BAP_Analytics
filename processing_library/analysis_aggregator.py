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
        if t.get('per_cycle', False):
            per_cycle_targets.append(t)
            
        elif t.get('time_series', False):
            time_series_targets.append(t)
            
        else:
            single_value_targets.append(t)

    print(f'Found {len(single_value_targets)} single-value targets')
    print(f'Found {len(per_cycle_targets)} per-cycle targets')
    print(f'Found {len(time_series_targets)} time-series targets')

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
    Stacks columns top-to-bottom without top padding. 
    Each column is appended in order from the list of DFs.
    Missing columns => NaN. Rows won't align across columns.
    """
    if not dataframes:
        return pd.DataFrame()

    all_cols = set()
    for df in dataframes:
        all_cols.update(df.columns)
    all_cols = sorted(all_cols)

    col_data = {}
    for col in all_cols:
        collected_vals = []
        for df in dataframes:
            if col in df.columns:
                collected_vals.extend(df[col].tolist())
        col_data[col] = collected_vals

    max_len = max(len(vals) for vals in col_data.values()) if col_data else 0
    for col, vals in col_data.items():
        short_by = max_len - len(vals)
        if short_by > 0:
            vals.extend([np.nan] * short_by)

    final_df = pd.DataFrame(col_data)
    return final_df


def _build_single_value_df(groups, targets, c_rate_tol):
    """
    Builds a wide DataFrame where each single-value target key is its own column.
    Rows won't align across columns. Each column is appended top-to-bottom.
    """
    if not targets:
        return pd.DataFrame()

    all_keys = [t["key"] for t in targets]
    col_data = {k: [] for k in all_keys}

    for tgt in targets:
        tkey = tgt["key"]
        
        ivar = tgt.get("interest_variable")
        ignore_keys = {'key','interest_variable','per_cycle','aggregation','time_series'}
        crit_keys  = set(tgt.keys()) - ignore_keys
        i = 0
        matches_for_this_target = []
        for grp in groups:
            i += 1
            match = True
            for c in crit_keys:
                tval = tgt[c]
                gval = grp.get(c, 0)  # direct dictionary key
                if c == 'crate':
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
                    if gval != tval:
                         
                        match = False
                        break
            if match:
                val = grp.get(ivar, np.nan)
       
                # print('target match',tgt)
                matches_for_this_target.append(val)

        col_data[tkey].extend(matches_for_this_target)

    # Pad columns to same length
    max_len = max(len(vals) for vals in col_data.values()) if col_data else 0
    for k, vals in col_data.items():
        diff = max_len - len(vals)
        if diff > 0:
            vals.extend([np.nan]*diff)

    return pd.DataFrame(col_data, columns=all_keys)


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
        agg  = tgt.get("aggregation", None)

        ignore_keys = {'key','interest_variable','per_cycle','aggregation','time_series'}
        crit_keys = set(tgt.keys()) - ignore_keys

        for grp in groups:
            cyc = grp.get("cycle")
            if cyc is None:
                continue
            match = True
            for c in crit_keys:
                tval = tgt[c]
                gval = grp.get(c, None)
                if c == 'c_rate':
                    if gval is None or abs(gval - tval) > c_rate_tol:
                        match = False
                        break
                elif c == 'range_c_rate':
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
    Each matched group => sub-slice from raw_data is appended top-to-bottom.
    """
    if not targets or raw_data is None:
        return pd.DataFrame()

    all_keys = [t["key"] for t in targets]
    column_data = {k: [] for k in all_keys}

    for tgt in targets:
        tkey = tgt["key"]
        ivar = tgt.get("interest_variable")

        ignore_keys= {'key','interest_variable','per_cycle','aggregation','time_series'}
        crit_keys= set(tgt.keys()) - ignore_keys

        matched_group= None
        for grp in groups:
            match = True
            for c in crit_keys:
                tval = tgt[c]
                gval = grp.get(c, None)
                if c == 'crate':
                    if len(tval) == 1:
                        if abs(gval - tval) > c_rate_tol:
                            match = False
                            break
                    else:
                        lo, hi = tval
                        if gval is None or not (lo <= gval <= hi):
                            match = False
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
                    if "voltage" in sub_df.columns:
                        values = sub_df["voltage"]
                    else:
                        values = pd.Series([np.nan]*(e_i - s_i +1), index=sub_df.index)

                for val in values:
                    row_dict = {col: np.nan for col in all_keys}
                    row_dict[tkey] = val
                    for colk in all_keys:
                        column_data[colk].append(row_dict[colk])
            else:
                row_dict = {col: np.nan for col in all_keys}
                row_dict[tkey] = np.nan
                for colk in all_keys:
                    column_data[colk].append(row_dict[colk])
        else:
            row_dict = {col: np.nan for col in all_keys}
            row_dict[tkey] = np.nan
            for colk in all_keys:
                column_data[colk].append(row_dict[colk])

    max_len = max(len(lst) for lst in column_data.values()) if column_data else 0
    for colk, vlist in column_data.items():
        while len(vlist) < max_len:
            vlist.append(np.nan)

    df_ts = pd.DataFrame(column_data)
    return df_ts