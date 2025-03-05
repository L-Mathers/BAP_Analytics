import pandas as pd
import numpy as np

def combine_charge_discharge_with_shift_reset(
    df: pd.DataFrame,
    charge_col: str,
    discharge_col: str,
    voltage_col: str,
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
    print(vmin)

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
        print(step_type)    

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
            # Check the previous group’s end conditions
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

# Example usage:
df = pd.read_csv("/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/20240726_150811_Z61_EVE_C1_ZPg_D00_SN14524.csv")
df = pd.read_csv("/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/subset_first_100cycles.csv")
df = pd.read_csv("/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/BMW_LTF_2580_002_10SOC_Cycling_366_T35_26-06-2024_AllData.csv")

df = combine_charge_discharge_with_shift_reset(
    df,
    charge_col="Charge Capacity [Ah]",
    discharge_col="Discharge Capacity [Ah]",
    voltage_col="",
    combined_col="capacity",
    min_voltage_offset=0.2,
)
print(df.head())

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot combined capacity
axs[0].plot(df["capacity"], label="Combined Capacity")
axs[0].set_xlabel("Index")
axs[0].set_ylabel("Capacity")
axs[0].legend()

# Plot charge and discharge capacities
axs[1].plot(df["Ah-Ch-Set"], label="Charge Capacity")
axs[1].plot(df["Ah-Dis-Set"], label="Discharge Capacity")
axs[1].set_xlabel("Index")
axs[1].set_ylabel("Capacity (Ah)")
axs[1].legend()

# Plot voltage
axs[2].plot(df["U[V]"], label="Voltage", color="orange")
axs[2].set_xlabel("Index")
axs[2].set_ylabel("Voltage (V)")
axs[2].legend()

plt.tight_layout()
plt.show()
