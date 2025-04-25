def find_last_zero_current(df, end_index):
    # Iterate through the DataFrame starting from end_index
    # and find the last occurrence of zero in the "current" column
    possible_zero_idx = end_index
    for i in range(end_index + 1, len(df)):
        if df["current"].iloc[i] != 0:
            return possible_zero_idx
        else:
            possible_zero_idx = i
    return possible_zero_idx


def daimler_formula(v1, v2, v3, t1, t2, t3, I):
    # Interpolated OCV at time t2
    voc_t = ((v3 - v1) / (t3 - t1)) * t2 + ((v1 * t3 - v3 * t1) / (t3 - t1))

    # Internal resistance
    R = (v2 - voc_t) / I
    return R


def dcir_processing(df, start_index, end_index, pulse_time, client=None):

    if client == None:
        print("No client specified, using default processing.")
    if client == "daimler_truck":
        subset = df.iloc[start_index:end_index]
        I = subset["current"].median()
        v1 = df["voltage"].iloc[start_index - 1]
        t1 = df["time"].iloc[start_index - 1]
        i_relaxed = find_last_zero_current(df, end_index)
        v3 = df["voltage"].iloc[i_relaxed]
        t2 = df["time"].iloc[i_relaxed]
        t3 = df["time"].iloc[i_relaxed]
        print("pulse_time", pulse_time)
        print("first time", subset["time"].iloc[0])
        print("time subset", subset["time"])
        target_time = subset["time"].iloc[0] + pulse_time

        # Check if any row meets the condition
        if (subset["time"] >= target_time).any():
            sub_df = subset[subset["time"] >= target_time]
        else:
            sub_df = subset.iloc[[-1]]  # Select the last row as a DataFrame
            print("No time greater than target_time found in subset")
        v2 = sub_df["voltage"].iloc[0]
        t2 = sub_df["time"].iloc[0]
        # Calculate DCIR using the formula
        dcir = daimler_formula(v1, v2, v3, t1, t2, t3, I) * 1000

        return dcir

        last_zero_idx = find_last_zero_current(df, end_index + 1)
