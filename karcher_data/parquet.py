import os

import pandas as pd


def convert_parquets_to_csv(folder_path):
    """
    Convert all parquet files in the given folder to CSV files.

    Parameters:
    folder_path (str): The path to the folder containing parquet files.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".parquet"):
                print(f"Converting {file} to csv")
                parquet_file = os.path.join(root, file)
                csv_file = os.path.splitext(parquet_file)[0] + ".csv"
                try:
                    df = pd.read_parquet(parquet_file)
                    df.to_csv(csv_file, index=False)
                    print(f"Converted {parquet_file} to {csv_file}")
                except Exception as e:
                    print(f"Failed to convert {parquet_file}: {e}")


if __name__ == "__main__":
    folder_path = "/Users/liammathers/Desktop/Github/BAP_Analytics/karcher_data"
    convert_parquets_to_csv(folder_path)
