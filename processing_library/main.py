# main.py
import os
import sys

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
    path_to_static_dict = (
        "/Users/liammathers/Desktop/Github/BAP_Analytics/processing_library/Static_Dict.json"
    )
    path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/20240726_150811_Z61_EVE_C1_ZPg_D00_SN14524.csv"

    # path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/EV_SKE_556_019_RPT3_423_T25_15-07-2024_AllData.csv"
    # path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/harmonised_data.csv"
    path_to_data_csv = (
        "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/subset_first_100cycles.csv"
    )
    # path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/20240610_155644_Z61_EVE_C1_Zykl_D01_SN13711.mf4.csv"
    # path_to_data_csv = '/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/IV_GO_R115C_005_LP18_549_T25_13-12-2024_AllData.csv'
    path_to_data_csv = (
        "/Users/liammathers/Desktop/Github/BAP_Analytics/karcher_data/5_1993_AllData_part_1.csv"
    )

    # rpt
    path_to_data_csv = (
        "/Users/liammathers/Desktop/Github/BAP_Analytics/karcher_data/5_1970_AllData_part_1.csv"
    )

    path_to_data_csv = "/Users/liammathers/Desktop/Github/BAP_Analytics/Daimler/1723817650_31_1_DT_DCIR_0deg_part1_AllData.csv"

    # Load the static dictionary
    with open(path_to_static_dict, "r") as f:
        base_config = json.load(f)

    # Read CSV data
    df_input = pd.read_csv(path_to_data_csv)

    # Define test parameters
    combined_input = {
        "test_type": "Rate Performance Test",
        "cell_limits": {"capacity": 159},
        "user_input": {
            "pulse_durations": [1, 5, 10],
            "special_crates": [],
            "voltage_relaxation": False,
            "dcir_normalization": [],
        },
    }
    # combined_input = {
    #     "test_type": "Cycle Aging",
    #     "cell_limits": {"capacity": 3.5},
    #     "user_input": {
    #         "nominal_normalization": True,
    #         "first_cycle_normalization": True,
    #     },
    # }

    # combined_input = {
    #     "test_type": "Combined RPT/Cycling",
    #     "cell_limits": {"capacity": 159},
    #     "user_input": {
    #         "pulse_durations": [1, 5, 10],
    #         "special_crates": [],
    #         "voltage_relaxation": False,
    #         "dcir_normalization": [50, 1],
    #         "pocv": False,
    #         "nominal_normalization": True,
    #         "first_cycle_normalization": True,
    #     },
    # }

    # Process the test data
    # 4) Call the processing function
    results_df = process_lifetime_test(
        data=df_input, combined_input=combined_input, config=base_config
    )

    # Print results and save to file
    print(results_df.head(50))
    results_df.to_csv("analysis_results.csv", index=False)


if __name__ == "__main__":
    main()
