import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import table as pd_table
import datetime

def plot_metrics_boxplots(metrics):
        
        N = metrics.shape[0]

        precision_mean = metrics['precision'].mean()*100
        recall_mean = metrics['recall'].mean()*100
        f1_mean = metrics['f1_score'].mean()*100

        precision_median = metrics['precision'].median()*100
        recall_median = metrics['recall'].median()*100
        f1_median = metrics['f1_score'].median()*100

        fig, axes = plt.subplots(1, 3, figsize=(9, 4))

        plt.rcParams['font.family'] = 'Times New Roman'

        box_properties = {
            'vert': True,
            'notch': True,
            'patch_artist': True,
            'showmeans': True,
            'medianprops': dict(color='red', linewidth=2, linestyle='-'),
            'meanprops': dict(color='green',marker='v'),
            'boxprops': dict(facecolor='lightgrey', linestyle='-', linewidth=1, color='black'),
            'flierprops': dict(marker='o', markerfacecolor='black', markersize=4, linestyle='none', markeredgecolor='black') 
        }
        jitter = np.random.normal(loc=1.4, scale=0.02, size=N)

        # Precision boxplot
        axes[0].boxplot(metrics['precision']*100, **box_properties)
        axes[0].scatter(jitter, metrics['precision']*100, color='black', alpha=0.8, s=4)
        axes[0].set_title('Boxplot of precision scores', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('precision [%]', fontsize=12)
        """axes[0].annotate(f'median: {precision_median:.2f} %', fontsize=9, color='red',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.9), textcoords='axes fraction')"""
        axes[0].annotate(f'mean: {precision_mean:.2f} %', fontsize=9, color='green',
                        xy=(0.5, 0.5), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.5), textcoords='axes fraction')
        
        # Recall boxplot
        axes[1].boxplot(metrics['recall']*100, **box_properties)
        axes[1].scatter(jitter, metrics['recall']*100, color='black', alpha=0.8, s=4)
        axes[1].set_title('Boxplot of recall scores', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('recall score [%]', fontsize=12)
        """axes[1].annotate(f'median: {recall_median:.2f} %', fontsize=9, color='red',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.9), textcoords='axes fraction')"""
        axes[1].annotate(f'mean: {recall_mean:.2f} %', fontsize=9, color='green',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.85), textcoords='axes fraction')

        # F1 score boxplot
        axes[2].boxplot(metrics['f1_score']*100, **box_properties)
        axes[2].scatter(jitter, metrics['f1_score']*100, color='black', alpha=0.8, s=4)
        axes[2].set_title('Boxplot of F1 scores', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('F1 score [%]', fontsize=12)
        """axes[2].annotate(f'median: {f1_median:.2f} %', fontsize=9, color='red',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.9), textcoords='axes fraction')"""
        axes[2].annotate(f'mean: {f1_mean:.2f} %', fontsize=9, color='green',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.85), textcoords='axes fraction')

        for i in range(3):
            axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[i].set_xlim((0.8,1.5))
              
        legend_labels = ['median', 'mean']
        legend_handles = [
            plt.Line2D([0], [0], color='red', linewidth=3),   # median
            plt.Line2D([0], [0], color='green', marker='v', linestyle='none'),  # mean
        ]
        fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)

        plt.tight_layout()
        plt.show()


def plot_highlights_distribution(highlights_per_query):
      
    plt.figure(figsize=(4, 4))
    plt.rcParams['font.family'] = 'Times New Roman'

    counts, bins, patches = plt.hist(highlights_per_query, np.arange(min(highlights_per_query) - 0.5, max(highlights_per_query) + 1.5, 1), edgecolor='black', color='lightgrey')
    plt.xticks(range(min(highlights_per_query), max(highlights_per_query) + 1))
    plt.xlabel("# of highlights (per query)")
    plt.title("Distribution of Highlights per Query", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for count, patch in zip(counts, patches):
        plt.text(patch.get_x() + patch.get_width()/2, count, str(int(count)), ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_highlighted_tokens_distribution(tokens_per_highlight):
     
    plt.figure(figsize=(8, 4))
    plt.rcParams['font.family'] = 'Times New Roman'

    counts, bins, patches = plt.hist(tokens_per_highlight, edgecolor='black', color='lightgrey')
    plt.yticks(range(0, int(max(counts)) + 1))
    plt.xlabel("# of tokens (per highlight)")
    plt.title("Distribution of Tokens per Highlight", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    xmin = min(tokens_per_highlight)
    xmax = max(tokens_per_highlight)
    plt.axvline(x=xmin, color='red', linestyle='--', linewidth=1)
    plt.axvline(x=xmax, color='red', linestyle='--', linewidth=1)
    plt.text(xmin+1, plt.ylim()[1]*0.95, f'min: {xmin}', color='red', ha='left', fontsize=10, fontweight='bold')
    plt.text(xmax-1, plt.ylim()[1]*0.95, f'max: {xmax}', color='red', ha='right', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_results_table(results_str):

    fig, ax = plt.subplots(figsize=(8, 8))

    table = ax.table(cellText=results_str.values, colLabels=results_str.columns, loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#D3D3D3')  # Light grey color for header
        cell.set_text_props(ha='center', va='center')
        cell.set_height(0.05)

    ax.axis('off')


    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results_table_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

