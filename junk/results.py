import numpy as np
import pandas as pd

# Load the CSV file
file_path = "/Users/liammathers/Desktop/Github/BAP_Analytics/junk/results comparison - Sheet1.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Initialize dictionaries and lists to store results
percent_differences = {}
all_differences = []

# Iterate through columns two at a time
for i in range(0, len(df.columns) - 1, 2):
    calculated_col = df.columns[i]
    actual_col = df.columns[i + 1]

    # Only proceed if the second column has no name (actual values)
    if pd.isna(actual_col):
        calc_vals = df[calculated_col]
        actual_vals = df.iloc[:, i + 1]

        # Avoid division by zero or invalid operations
        with np.errstate(divide="ignore", invalid="ignore"):
            percent_diff = (
                np.abs(calc_vals - actual_vals)
                / np.where(actual_vals != 0, actual_vals, np.nan)
                * 100
            )

        # Save average percent difference for the header
        avg_diff = percent_diff.mean(skipna=True)
        percent_differences[calculated_col] = avg_diff

        # Collect all individual percent differences
        all_differences.extend(percent_diff.dropna().tolist())

# Calculate overall average percent difference
overall_average = np.mean(all_differences)

# Output results
print("Average Percent Difference per Header:")
for header, avg in percent_differences.items():
    print(f"{header}: {avg:.2f}%")

print(f"\nOverall Average Percent Difference: {overall_average:.2f}%")
