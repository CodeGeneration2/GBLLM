import pandas as pd
import os
import subprocess
import ast
from pathlib import Path
import re



# #####################################################################################################################🔖💡✅🟨
DEBUG = False
DATASET_PATH = r""


# #####################################################################################################################🔖💡✅🟨
def main_process():
    df = pd.read_csv(DATASET_PATH)
    print("✅✅✅ Initial data row count:", len(df))

    column_names = df.columns.tolist()


    # Assuming df is your DataFrame
    # Copy 'Slow_Code' column to 'input' column
    repo_unique_ids = df['Repo_Unique_ID'].tolist()
    original_code_times = df['input__time(us)'].tolist()
    
    # Note: These column keys have been translated. Ensure they match your CSV headers.
    medium_code_times = df['Cot_ShortNL_CFG_SlowMidFast_Temp13_Round4_Sorted_MidTime'].tolist()
    fast_code_times = df['Cot_ShortNL_CFG_SlowMidFast_Temp13_Round4_Sorted_FastTime'].tolist()

    for i in range(len(original_code_times)):
        if medium_code_times[i] == 'pass':
            continue
        if fast_code_times[i] == 'pass':
            fastest_time = medium_code_times[i]
        else:
            fastest_time = min(float(medium_code_times[i]), float(fast_code_times[i]))


        if float(fastest_time) > 0.9 * float(original_code_times[i]):
            continue
        
        
        print(f"\n\n💡💡💡 {i} Repo Unique ID: {repo_unique_ids[i]}, \nOriginal Code Time: {original_code_times[i]} us")
        if medium_code_times[i] != 'pass':
            print(f"Medium Code Time: {medium_code_times[i]} us")
            speedup_ratio = 100 * (float(original_code_times[i]) / float(medium_code_times[i]))
            print(f"✅✅✅ Medium Code Speedup Ratio: {speedup_ratio:.2f} x")
        if fast_code_times[i] != 'pass':
            print(f"Fast Code Time: {fast_code_times[i]} us")
            speedup_ratio = 100 * (float(original_code_times[i]) / float(fast_code_times[i]))
            print(f"✅✅✅ Fast Code Speedup Ratio: {speedup_ratio:.2f} x")






    
    

# #################################################################################################################################################
if __name__ == "__main__":
    # Run main processing function
    main_process()