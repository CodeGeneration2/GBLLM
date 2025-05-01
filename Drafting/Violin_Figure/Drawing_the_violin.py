import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def main():
    ax1 = initialize_function()
    ax1 = plot_generation_function(ax1)
    post_processing_function(ax1)

def initialize_function():
    fig, ax1 = plt.subplots(figsize=(23, 7), layout='constrained')
    fig.subplots_adjust(wspace=0)  # Set the width space between subplots to 0

    # -----------------------
    ax1.set_ylim([0, 800])

    # Replace y-axis tick labels
    ax1.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800])  # Original tick positions
    ax1.set_yticklabels(["NC", "NO", " ", "NH", "  ", "FH", "FH-20%", "FH-30%", "FH-Max"])  # Replace with your labels

    return ax1

def plot_generation_function(ax1):
    with open(r"Full_Model_Data_Dictionary_Gen_CodeLlama.txt", 'r', encoding='UTF-8') as f:
        CodeLlama_Gen_dict = eval(f.read())
    with open(r"Full_Model_Data_Dictionary_Gen_Gemini.txt", 'r', encoding='UTF-8') as f:
        Gemini_Gen_dict = eval(f.read())
    with open(r"Full_Model_Data_Dictionary_Gen_GPT3.txt", 'r', encoding='UTF-8') as f:
        GPT3_Gen_dict = eval(f.read())
    with open(r"Full_Model_Data_Dictionary_Gen_GPT4.txt", 'r', encoding='UTF-8') as f:
        GPT4_Gen_dict = eval(f.read())

    with open(r"Full_Model_Data_Dictionary_Top_CodeLlama.txt", 'r', encoding='UTF-8') as f:
        CodeLlama_Top_dict = eval(f.read())
    with open(r"Full_Model_Data_Dictionary_Top_Gemini.txt", 'r', encoding='UTF-8') as f:
        Gemini_Top_dict = eval(f.read())
    with open(r"Full_Model_Data_Dictionary_Top_GPT3.txt", 'r', encoding='UTF-8') as f:
        GPT3_Top_dict = eval(f.read())
    with open(r"Full_Model_Data_Dictionary_Top_GPT4.txt", 'r', encoding='UTF-8') as f:
        GPT4_Top_dict = eval(f.read())

    CodeLlama_generate1 = CodeLlama_Gen_dict["generate1"]
    CodeLlama_generate3 = CodeLlama_Gen_dict["generate3"]
    CodeLlama_generate5 = CodeLlama_Gen_dict["generate5"]
    CodeLlama_5select1 = CodeLlama_Top_dict["5select1"]
    CodeLlama_5select3 = CodeLlama_Top_dict["5select3"]
    CodeLlama_5select5 = CodeLlama_Top_dict["5select5"]

    # GPT4_5select1
    Gemini_generate1 = Gemini_Gen_dict["generate1"]
    Gemini_generate3 = Gemini_Gen_dict["generate3"]
    Gemini_generate5 = Gemini_Gen_dict["generate5"]
    Gemini_5select1 = Gemini_Top_dict["5select1"]
    Gemini_5select3 = Gemini_Top_dict["5select3"]
    Gemini_5select5 = Gemini_Top_dict["5select5"]

    GPT3_generate1 = GPT3_Gen_dict["generate1"]
    GPT3_generate3 = GPT3_Gen_dict["generate3"]
    GPT3_generate5 = GPT3_Gen_dict["generate5"]
    GPT3_5select1 = GPT3_Top_dict["5select1"]
    GPT3_5select3 = GPT3_Top_dict["5select3"]
    GPT3_5select5 = GPT3_Top_dict["5select5"]

    GPT4_generate1 = GPT4_Gen_dict["generate1"]
    GPT4_generate3 = GPT4_Gen_dict["generate3"]
    GPT4_generate5 = GPT4_Gen_dict["generate5"]
    GPT4_5select1 = GPT4_Top_dict["5select1"]
    GPT4_5select3 = GPT4_Top_dict["5select3"]
    GPT4_5select5 = GPT4_Top_dict["5select5"]

    Time_CodeLlama  = np.array(CodeLlama_generate1 + CodeLlama_generate3 + CodeLlama_generate5 + CodeLlama_5select3 + CodeLlama_5select5)
    Time_Gemini     = np.array(Gemini_generate1 + Gemini_generate3 + Gemini_generate5 + Gemini_5select3 + Gemini_5select5)
    Time_GPT3       = np.array(GPT3_generate1 + GPT3_generate3 + GPT3_generate5 + GPT3_5select3 + GPT3_5select5)
    Time_GPT4       = np.array(GPT4_generate1 + GPT4_generate3 + GPT4_generate5 + GPT4_5select3 + GPT4_5select5)

    CodeLlama   = np.array(['1@1'] * len(CodeLlama_generate1) + ['3@1'] * len(CodeLlama_generate3) + ['5@1'] * len(CodeLlama_generate5) + ['5@3'] * len(CodeLlama_5select3) + ['5@5'] * len(CodeLlama_5select5))
    Gemini      = np.array(['1@1'] * len(Gemini_generate1) + ['3@1'] * len(Gemini_generate3) + ['5@1'] * len(Gemini_generate5) + ['5@3'] * len(Gemini_5select3) + ['5@5'] * len(Gemini_5select5))
    GPT3        = np.array(['1@1'] * len(GPT3_generate1) + ['3@1'] * len(GPT3_generate3) + ['5@1'] * len(GPT3_generate5) + ['5@3'] * len(GPT3_5select3) + ['5@5'] * len(GPT3_5select5)) 
    GPT4        = np.array(['1@1'] * len(GPT4_generate1) + ['3@1'] * len(GPT4_generate3) + ['5@1'] * len(GPT4_generate5) + ['5@3'] * len(GPT4_5select3) + ['5@5'] * len(GPT4_5select5)) 

    # We can put the data in a pandas dataframe
    df = pd.DataFrame({
        'LLMs':  ['CodeLlama']   * (len(CodeLlama_generate1) + len(CodeLlama_generate3) + len(CodeLlama_generate5) + len(CodeLlama_5select3) + len(CodeLlama_5select5))
                +    ['Gemini']      * (len(Gemini_generate1) + len(Gemini_generate3) + len(Gemini_generate5) + len(Gemini_5select3) + len(Gemini_5select5))
                +    ['ChatGPT']     * (len(GPT3_generate1) + len(GPT3_generate3) + len(GPT3_generate5) + len(GPT3_5select3) + len(GPT3_5select5))
                +    ['GPT4']        * (len(GPT4_generate1) + len(GPT4_generate3) + len(GPT4_generate5) + len(GPT4_5select3) + len(GPT4_5select5)),
        'Performance Level': np.concatenate([Time_CodeLlama, Time_Gemini, Time_GPT3, Time_GPT4]),
        'Metric': np.concatenate([CodeLlama, Gemini, GPT3, GPT4])
    })

    # Use seaborn's violinplot method to draw a violin plot
    sns.violinplot(x="LLMs", y="Performance Level", hue="Metric", data=df, ax=ax1, cut=0, palette=["#8ECFC9", "#F7E1ED", "#96C37D", "#FFBE7A", "#BB9727"], scale="area")

    return ax1

def post_processing_function(ax1):

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    # Bold tick marks
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)

    # Bold tick labels and axis labels
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)

    ax1.tick_params(axis='x', labelsize=15)  # Set x-axis tick font size to 15
    ax1.tick_params(axis='y', labelsize=15)  # Set y-axis tick font size to 15

    ax1.set_xlabel('LLMs', fontsize=20, fontweight='bold', fontdict={'family': 'Arial'})
    ax1.set_ylabel('Performance Level', fontsize=20, fontweight='bold', fontdict={'family': 'Arial'})

    # ------------------------------------#
    ax1.grid(which="major", axis='y')
    ax1.set_axisbelow(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
