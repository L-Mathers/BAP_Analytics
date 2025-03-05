# main.py
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd
from processing_library.processing import process_lifetime_test

def main():
    """
    Load data, process it using `process_lifetime_test`, and save results.
    """
    # Paths to JSON static dictionary and CSV test data
    path_to_static_dict = "/Users/liammathers/Desktop/Github/BAP_Analytics/processing_library/Static_Dict.json"
    path_to_data_csv    = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/20240726_150811_Z61_EVE_C1_ZPg_D00_SN14524.csv"
    path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/EV_SKE_556_019_RPT3_423_T25_15-07-2024_AllData.csv"
    path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/harmonised_data.csv"
    path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/subset_first_100cycles.csv"
    path_to_data_csv = "/Users/liammathers/Desktop/Github/bmw_lifetime_processing_int/local_test/events/input/upload/failed_to_process/Z61/20240701_110753_Z61_EVE_C1_Zykl_D01_SN14505.csv.csv"
# 
    # Load the static dictionary
    with open(path_to_static_dict, "r") as f:
        base_config = json.load(f)

    # Read CSV data
    df_input = pd.read_csv(path_to_data_csv)
    # Define test parameters
    combined_input = {
        "test_type": "Rate Performance Test",
        "cell_limits": {"capacity": 11},
        "user_input": {
            "pulse_durations": [1, 2],
            "special_crates": [1, 2],
            "voltage_relaxation": False
        }
    }

    # Process the test data
    # 4) Call the processing function
    results_df = process_lifetime_test(
        data=df_input,
        combined_input=combined_input,
        config=base_config
    )

    # Print results and save to file
    print(results_df.head(20))
    results_df.to_csv("analysis_results.csv", index=False)

if __name__ == "__main__":
    main()