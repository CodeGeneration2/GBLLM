import matplotlib.pyplot as plt
import numpy as np

def plot_example_chart():
    # Data
    categories = ['Instruction', 'ICL', 'RAG', 'COT', 'SBLLM', 'GBLLM']

    # New Version
    NC = [29, 27.18, 35.25, 44.11, 23.06, 4.4328]
    NO = [66, 62.94, 52.25, 31.11, 40.06, 12.9074]
    NH = [ 4,  7.94, 10.25, 19.11, 28.06, 66.8839]
    FH = [ 1,  1.94,  2.25, 5.67,  8.82,  15.7757]

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 4))

    # Plot four groups of bars
    ax.bar(x - 1.5*width, NC, width, label='NC', color='#FE7E0D', hatch='\\\\\\\\')
    ax.bar(x - 0.5*width, NO, width, label='NO', color='#66CC66', hatch='||||')
    ax.bar(x + 0.5*width, NH, width, label='NH', color='#8C564B', hatch='*')
    ax.bar(x + 1.5*width, FH, width, label='FH', color='#1BA1E2', hatch='/////')

    # Coordinate and style settings
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_ylim(0, 80)
    ax.set_ylabel('Percentage (%)', fontsize=15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Keep outer border
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # ax.legend(frameon=False, fontsize=10, loc='upper right')
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_example_chart()
