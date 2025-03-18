#!/usr/bin/env python
# test_soc_estimation.py

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from processing_library.feature_extraction import estimate_soc
import argparse

def main():
    """
    Test the improved SOC estimation using charge and discharge capacity columns.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test SOC estimation with charge/discharge capacity columns')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional plots and CSV exports')
    parser.add_argument('--data', type=str, default="/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/20240726_150811_Z61_EVE_C1_ZPg_D00_SN14524.csv", 
                        help='Path to the CSV data file')
    parser.add_argument('--use-capacity', action='store_true', 
                        help='Test using only the capacity column (ignoring charge/discharge capacity columns)')
    args = parser.parse_args()
    
    # Path to test data CSV
    path_to_data_csv = args.data
    
    # Read CSV data
    print(f"Loading data from {path_to_data_csv}")
    df_input = pd.read_csv(path_to_data_csv)
    
    if args.debug:
        # Print column names to verify charge and discharge capacity columns
        print("Available columns:")
        for col in df_input.columns:
            print(f"  - {col}")
    
    # Rename columns to match expected names
    column_mapping = {
        "Test Time (s)": "time",
        "Current (A)": "current",
        "Voltage (V)": "voltage",
        "Capacity (Ah)": "capacity",
        "Charge Capacity (Ah)": "charge_capacity",
        "Discharge Capacity (Ah)": "discharge_capacity",
        "Charge Energy (Wh)": "charge_energy",
        "Discharge Energy (Wh)": "discharge_energy"
    }
    
    df_input = df_input.rename(columns=column_mapping)
    
    # If testing with only capacity column, remove charge/discharge capacity columns
    if args.use_capacity:
        print("Testing with only the capacity column (ignoring charge/discharge capacity columns)")
        if 'charge_capacity' in df_input.columns:
            df_input = df_input.drop(columns=['charge_capacity'])
        if 'discharge_capacity' in df_input.columns:
            df_input = df_input.drop(columns=['discharge_capacity'])
    
    # Add Step Type column if not present (required for SOC estimation)
    if "Step Type" not in df_input.columns:
        print("Adding Step Type column based on current values")
        df_input["Step Type"] = "rest"
        df_input.loc[df_input["current"] > 0.1, "Step Type"] = "charge"
        df_input.loc[df_input["current"] < -0.1, "Step Type"] = "discharge"
    
    # Run SOC estimation
    print("Running SOC estimation...")
    df_result = estimate_soc(
        df=df_input,
        voltage_col="voltage",
        step_col="Step Type",
        capacity_col="capacity",
        charge_capacity_col="charge_capacity",
        discharge_capacity_col="discharge_capacity",
        voltage_tolerance=0.1,
        update_nominal=False,
        nom_cap=32  # Set to the nominal capacity of the cell
    )
    
    # Create debug plots and exports if debug mode is enabled
    if args.debug:
        # Add debugging for capacity resets
        if 'charge_capacity' in df_result.columns:
            df_result['charge_cap_diff'] = df_result['charge_capacity'].diff().fillna(0)
            charge_reset_mask = df_result['charge_capacity'].abs() < df_result['charge_capacity'].shift(1).abs()
            charge_resets = charge_reset_mask.sum()
        else:
            charge_resets = 0
            
        if 'discharge_capacity' in df_result.columns:
            df_result['discharge_cap_diff'] = df_result['discharge_capacity'].diff().fillna(0)
            discharge_reset_mask = df_result['discharge_capacity'].abs() < df_result['discharge_capacity'].shift(1).abs()
            discharge_resets = discharge_reset_mask.sum()
        else:
            discharge_resets = 0
            
        if 'capacity' in df_result.columns:
            df_result['capacity_diff'] = df_result['capacity'].diff().fillna(0)
            capacity_reset_mask = df_result['capacity'].abs() < df_result['capacity'].shift(1).abs()
            capacity_resets = capacity_reset_mask.sum()
            print(f"Detected {capacity_resets} capacity resets")
        
        # Mark resets
        df_result['reset_marker'] = 0
        if 'charge_capacity' in df_result.columns and 'discharge_capacity' in df_result.columns:
            df_result.loc[charge_reset_mask | discharge_reset_mask, 'reset_marker'] = 1
            print(f"Detected {charge_resets} charge capacity resets and {discharge_resets} discharge capacity resets")
        elif 'capacity' in df_result.columns:
            df_result.loc[capacity_reset_mask, 'reset_marker'] = 1
        
        # Export data for debugging
        df_result.to_csv("debug_soc_estimation.csv", index=False)
        
        # Create debug plots
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Raw capacity values
        plt.subplot(4, 1, 1)
        if 'charge_capacity' in df_result.columns:
            plt.plot(df_result['time'], df_result['charge_capacity'], 'b-', label='Charge Capacity')
        if 'discharge_capacity' in df_result.columns:
            plt.plot(df_result['time'], df_result['discharge_capacity'], 'r-', label='Discharge Capacity')
        if 'capacity' in df_result.columns:
            plt.plot(df_result['time'], df_result['capacity'], 'g-', label='Capacity')
        
        # Mark resets with vertical lines
        reset_times = df_result.loc[df_result['reset_marker'] == 1, 'time']
        for t in reset_times:
            plt.axvline(x=t, color='k', linestyle='--', alpha=0.3)
        plt.legend()
        plt.title('Raw Capacity Values with Reset Points Marked')
        
        # Plot 2: Capacity differences
        plt.subplot(4, 1, 2)
        if 'charge_cap_diff' in df_result.columns:
            plt.plot(df_result['time'], df_result['charge_cap_diff'], 'b-', label='Charge Capacity Diff')
        if 'discharge_cap_diff' in df_result.columns:
            plt.plot(df_result['time'], df_result['discharge_cap_diff'], 'r-', label='Discharge Capacity Diff')
        if 'capacity_diff' in df_result.columns:
            plt.plot(df_result['time'], df_result['capacity_diff'], 'g-', label='Capacity Diff')
        plt.legend()
        plt.title('Capacity Differences')
        
        # Plot 3: Current
        plt.subplot(4, 1, 3)
        plt.plot(df_result['time'], df_result['current'], 'g-', label='Current')
        plt.legend()
        plt.title('Current Profile')
        
        # Plot 4: SOC
        plt.subplot(4, 1, 4)
        plt.plot(df_result['time'], df_result['soc'], 'k-', label='Estimated SOC')
        plt.legend()
        plt.title('Estimated State of Charge')
        
        plt.tight_layout()
        plt.savefig('debug_capacity_analysis.png', dpi=300)
        print("Debug plots saved to debug_capacity_analysis.png")
    
    # Always create the main SOC plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: SOC vs Time
    plt.subplot(3, 1, 1)
    plt.plot(df_result["time"], df_result["soc"], 'b-', label='Estimated SOC')
    plt.ylabel('SOC (%)')
    plt.title('Estimated State of Charge')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Accumulated Capacity vs Time
    plt.subplot(3, 1, 2)
    plt.plot(df_result["time"], df_result["Accumulated Capacity"], 'g-', label='Accumulated Capacity')
    plt.ylabel('Capacity (Ah)')
    plt.title('Accumulated Capacity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Current vs Time
    plt.subplot(3, 1, 3)
    plt.plot(df_result["time"], df_result["current"], 'r-', label='Current')
    plt.ylabel('Current (A)')
    plt.xlabel('Time (s)')
    plt.title('Current Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Add suffix to output files based on mode
    suffix = "_capacity_only" if args.use_capacity else ""
    plt.savefig(f'soc_estimation_results{suffix}.png', dpi=300)
    print(f"Results saved to soc_estimation_results{suffix}.png")
    
    # Save processed data
    df_result.to_csv(f"soc_estimation_results{suffix}.csv", index=False)
    print(f"Processed data saved to soc_estimation_results{suffix}.csv")

if __name__ == "__main__":
    main() 