import pandas as pd
from battery_analysis.main import process_lifetime_test

df_rpt = pd.read_csv("/Users/liammathers/Desktop/Github/BAP_Analytics/Testing/20240726_150811_Z61_EVE_C1_ZPg_D00_SN14524.csv")

combined_input_rpt = {
    "test_type": "Rate Performance Test",
    "cell_limits": {"capacity": 57.5},
    "user_input": {
        "pulse_durations": [1, 2],
        "special_crates": [1, 2],
        "voltage_relaxation": True
    }
}

results_rpt = process_lifetime_test(
    data=df_rpt,
    combined_input=combined_input_rpt,
    base_config=STATIC_DICT  # or load from JSON
)
print(results_rpt)