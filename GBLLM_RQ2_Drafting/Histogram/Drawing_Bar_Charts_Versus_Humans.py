import matplotlib.pyplot as plt
import numpy as np


def plot_example_chart():
    # Data
    categories = ['Instruction', 'ICL', 'RAG', 'COT', 'SBLLM', 'GBLLM']

    
    # New version    PIE - C++
    PIE_Cpp_NC = [20.79, 20.22, 26.12, 30.9, 19.8, 15.45]
    PIE_Cpp_NO = [21.21, 23.17, 16.99, 20.79, 20.51, 7.3]
    PIE_Cpp_NH = [38.76, 36.8, 34.97, 30.62, 35.53, 12.78]
    PIE_Cpp_FH = [19.24, 19.8, 21.91, 17.7, 24.16, 64.47]


    # PIE - Python
    PIE_Py_NC = [12.2, 11.38, 12.52, 18.86, 32.52, 6.67]
    PIE_Py_NO = [18.86, 11.06, 17.07, 19.35, 9.92, 9.92]
    PIE_Py_NH = [57.4, 67.97, 60.16, 51.22, 41.79, 58.7]
    PIE_Py_FH = [11.54, 9.59, 10.24, 10.57, 15.77, 24.72]


    x = np.arange(len(categories))
    width = 0.2

    # fig, ax = plt.subplots(figsize=(11, 4))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 3.7), sharey=True)

    # Plot four groups of bars (Subplot 1)
    ax1.bar(x - 1.5*width, PIE_Cpp_NC, width, label='NC', color='#FE7E0D', hatch='\\\\\\\\')
    ax1.bar(x - 0.5*width, PIE_Cpp_NO, width, label='NO', color='#1BA1E2', hatch='||||')
    ax1.bar(x + 0.5*width, PIE_Cpp_NH, width, label='NH', color='#8C564B', hatch='*')
    ax1.bar(x + 1.5*width, PIE_Cpp_FH, width, label='FH', color='#66CC66', hatch='/////')

    # Plot four groups of bars (Subplot 2)
    ax2.bar(x - 1.5*width, PIE_Py_NC, width, label='NC', color='#FE7E0D', hatch='\\\\\\\\')
    ax2.bar(x - 0.5*width, PIE_Py_NO, width, label='NO', color='#1BA1E2', hatch='||||')
    ax2.bar(x + 0.5*width, PIE_Py_NH, width, label='NH', color='#8C564B', hatch='*')
    ax2.bar(x + 1.5*width, PIE_Py_FH, width, label='FH', color='#66CC66', hatch='/////')
    

    # Coordinates and style settings
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=15)
    ax1.set_ylim(0, 70)
    ax1.set_ylabel('Percentage (%)', fontsize=15)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=15)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)


    # Configure spines (borders)
    ax1.spines['top'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    ax1.spines['right'].set_visible(False)

    ax2.spines['top'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(True)


    # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    # ax.legend(frameon=False, fontsize=10, loc='upper right')
    ax1.legend(frameon=False, loc='best')
    # ax2.legend(frameon=False, loc='upper left')
    ax2.legend(frameon=False, loc='best')


    plt.tight_layout(rect=[0, 0.05, 1, 1]) 


    # Place subplot descriptions at the bottom of the entire figure
    fig.text(0.25, 0.02,
             '(a) The proportion of different optimization level on C++ (O3).',
             ha='center', fontsize=14)
    fig.text(0.75, 0.02,
             '(b) The proportion of different optimization level on Python.',
             ha='center', fontsize=14)
    

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_example_chart()