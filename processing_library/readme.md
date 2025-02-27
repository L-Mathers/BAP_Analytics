# Battery Analysis Pipeline

## Overview
This project is designed to process battery test data efficiently. The workflow involves:
1. **Loading raw battery test data** (CSV format)
2. **Configuring analysis parameters** using a static configuration file
3. **Processing the data** (detecting charge/discharge cycles, grouping events, calculating efficiencies, etc.)
4. **Generating key performance indicators (KPIs)** from the processed data
5. **Saving the results** in a structured CSV file for further analysis

## Project Structure
```
battery_analysis/
├── __init__.py
├── config_builder.py       # Handles configuration and dynamic target construction
├── data_processing.py      # Prepares and cleans raw battery data (grouping, labeling, CV detection, etc.)
├── feature_extraction.py   # Extracts cycle information, OCV-SOC mapping, and efficiency metrics
├── analysis_aggregator.py  # Aggregates computed metrics into a structured format
├── processing.py           # Main processing function that orchestrates all steps
├── main.py                 # Entry point script for running the analysis
└── tests/                  # Unit tests for each module
```

## Workflow
### 1. **main.py** (Entry Point)
- Loads the static dictionary (`static_dict.json`)
- Reads the CSV file containing raw battery test data
- Defines the test parameters (e.g., test type, cell capacity, special conditions)
- Calls `process_lifetime_test()` from `processing.py`
- Saves the processed data into a results file (`analysis_results.csv`)

**Run this script with:**
```bash
python main.py 
```

### 2. **processing.py** (Main Processing Logic)
- Calls necessary functions from other modules to:
  - Match columns in raw data (`match_required_columns`)
  - Process and group test cycles (`psuedo_limit_extraction`, `add_cv_steps`)
  - Assign cycles and compute efficiencies (`assign_cycle_keys`, `calculate_coulombic_and_energy_eff`)
  - Aggregate results (`find_parameters_for_section`)
- Returns the final structured DataFrame
- This function can be called from `main.py` or any other script

### 3. **config_builder.py** (Configuration Management)
- Loads and processes the static dictionary (`static_dict.json`)
- Defines test-specific targets dynamically
- Ensures the analysis uses the correct parameters for each test type

### 4. **data_processing.py** (Data Cleaning & Preprocessing)
- Matches column names from various test data formats
- Identifies charge, discharge, and rest cycles
- Detects constant voltage (CV) phases and filters noisy data

### 5. **feature_extraction.py** (Cycle and Efficiency Analysis)
- Computes cycle-related parameters (capacity, energy, voltage trends)
- Maps open circuit voltage (OCV) to state of charge (SOC)
- Assigns cycle numbers based on charge-discharge pairs
- Computes efficiencies such as Coulombic efficiency and energy efficiency

### 6. **analysis_aggregator.py** (Final KPI Computation)
- Aggregates extracted metrics into structured output
- Formats output into a DataFrame for further analysis
- Ensures all required test targets are included in the results

### 7. **tests/** (Unit Testing)
- Contains pytest test cases for all key functions
- Tests include:
  - Data preprocessing and grouping validation
  - Feature extraction accuracy
  - Cycle identification correctness
  - End-to-end pipeline integrity

Run tests with:
```bash
pytest tests/
```

## How to Use the Processing Function in Other Scripts
Instead of running `main.py`, you can call `process_lifetime_test()` from another script:
```python
import pandas as pd
import json
from battery_analysis.processing import process_lifetime_test

# Load config
with open("/path/to/static_dict.json", "r") as f:
    base_config = json.load(f)

# Load data
csv_path = "/path/to/test_data.csv"
df_input = pd.read_csv(csv_path)

# Define test parameters
combined_input = {
    "test_type": "Cycle Aging",
    "cell_limits": {"capacity": 4.2},
    "user_input": {"normalize_by_first_cycle": False}
}

# Process data
results_df = process_lifetime_test(data=df_input, combined_input=combined_input, base_config=base_config)

# Save results
results_df.to_csv("custom_results.csv", index=False)
```

## Summary
- **Use `main.py`** for standard analysis workflows
- **Use `processing.py`** when integrating into other scripts
- **Use `pytest`** to verify functionality before deploying changes
- **All processing logic is modular** and can be customized for different battery tests

This structure ensures **modularity, reusability, and easy debugging** for battery data processing. 🚀

