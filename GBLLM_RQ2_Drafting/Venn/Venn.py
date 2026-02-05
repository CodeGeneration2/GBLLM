# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨

import sys
import re
import os
from tqdm import tqdm
import io
import tokenize
import keyword
import pandas as pd
import gc
import statistics
import pprint
import statistics
import matplotlib.pyplot as plt
from venn import venn
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles



print('\033[0:33m======================= Come on, Let\'s do this! ==============================\033[m')

# #####################################################################################################################🔖💡✅🟨❌
DEBUG = False

# Note: Please ensure the directory structure matches this English path
DATASET_ROOT_PATH = r"715__Paper_Figures_Code\Section_7.2.2_Venn_Diagram\Data"

# Create the first group of data
TOTAL_DATA_DICT = {
    "PIE_Cpp": {    'direct': set(),
                    'icl': set(),
                    'rag': set(),
                    'cot': set(),
                    'sbllm': set(),
                    'GBLLM': set(),     },

    "PIE_Py":{      'direct': set(),
                    'icl': set(),
                    'rag': set(),
                    'cot': set(),
                    'sbllm': set(),
                    'GBLLM': set(),     },

    "DB_Py": {      'direct': set(),
                    'icl': set(),
                    'rag': set(),
                    'cot': set(),
                    'sbllm': set(),
                    'GBLLM': set(),     },
}

# #####################################################################################################################🔖💡✅🟨❌
def main():

    filename_list = os.listdir(DATASET_ROOT_PATH)
    for filename in filename_list:

        if filename.startswith('PIE_Cpp'):
            test_type = "PIE_Cpp"
        elif filename.startswith('PIE_Py'):
            test_type = "PIE_Py"
        elif filename.startswith('DB_Py'):
            test_type = "DB_Py"
        
        # Extract method name from filename
        method_name = filename.split('_DeepSeekV32_')[-1].split('_')[0]
        
        with open(rf"{DATASET_ROOT_PATH}\{filename}", 'r', encoding='UTF-8') as f:
            # Note: The key in the input file dictionary is assumed to be translated as well. 
            # If the source data still has Chinese keys, please update this string to match the source.
            TOTAL_DATA_DICT[test_type][method_name] = set(eval(f.read())["GBLLM_idx_faster_than_human_rate"])

    # ------------------------------------------
    del TOTAL_DATA_DICT["PIE_Cpp"]["icl"]
    del TOTAL_DATA_DICT["PIE_Py"]["icl"]
    del TOTAL_DATA_DICT["DB_Py"]["icl"]


    # Create figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot the first Venn diagram on the first subplot - Key: add ax=ax1 parameter
    venn(TOTAL_DATA_DICT["PIE_Cpp"], ax=ax1)  # Explicitly specify drawing on ax1
    # Plot the second Venn diagram on the second subplot - Key: add ax=ax2 parameter
    venn(TOTAL_DATA_DICT["PIE_Py"], ax=ax2)  # Explicitly specify drawing on ax2


    # Key point: transform=ax.transAxes indicates using "Axes Coordinate System" (0~1)
    # Add text at the five corners
    five_corner_positions = [     (0.06, 0.7),  # Top left
                        (0.92, 0.7),  # Top right
                        (0.2, 0.06),  # Bottom left
                        (0.80, 0.06),  # Bottom right
                        (0.51, 0.95)    # Top center
                    ]

    # Description text for the five corners
    five_corner_labels = [  "Instruct",
                        "COT", 
                        "GBLLM",
                        "SBLLM",
                        "RAG"
                    ]
    for (x, y), label in zip(five_corner_positions, five_corner_labels):
        ax1.text(x, y, label, transform=ax1.transAxes, ha="center", va="center", fontsize=17, fontweight="bold")
        ax2.text(x, y, label, transform=ax2.transAxes, ha="center", va="center", fontsize=17, fontweight="bold")



    # Set titles
    # ax1.set_title("Experiment Group A", fontsize=14, fontweight='bold')
    # ax2.set_title("Experiment Group B", fontsize=14, fontweight='bold')


    # Add subtitles
    ax1.text(0.5, -0.02, "(a) C++ (O3).", 
            transform=ax1.transAxes, 
            ha='center', 
            fontsize=20, 
            )
    # Add subtitles
    ax2.text(0.5, -0.02, "(b) Python.", 
            transform=ax2.transAxes, 
            ha='center', 
            fontsize=20, 
            )


    # Remove legends
    ax1.legend_.remove()  
    ax2.legend_.remove()  

    # Adjust layout
    plt.tight_layout()
    plt.show()


# #################################################################################################################################################❌
if __name__ == '__main__':
    main()