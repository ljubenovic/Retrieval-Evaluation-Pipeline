import matplotlib.pyplot as plt
import numpy as np

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
        axes[0].annotate(f'median: {precision_median:.2f} %', fontsize=9, color='red',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.9), textcoords='axes fraction')
        axes[0].annotate(f'mean: {precision_mean:.2f} %', fontsize=9, color='green',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.85), textcoords='axes fraction')
        
        # Recall boxplot
        axes[1].boxplot(metrics['recall']*100, **box_properties)
        axes[1].scatter(jitter, metrics['recall']*100, color='black', alpha=0.8, s=4)
        axes[1].set_title('Boxplot of recall scores', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('recall score [%]', fontsize=12)
        axes[1].annotate(f'median: {recall_median:.2f} %', fontsize=9, color='red',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.9), textcoords='axes fraction')
        axes[1].annotate(f'mean: {recall_mean:.2f} %', fontsize=9, color='green',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.85), textcoords='axes fraction')

        # F1 score boxplot
        axes[2].boxplot(metrics['f1_score']*100, **box_properties)
        axes[2].scatter(jitter, metrics['f1_score']*100, color='black', alpha=0.8, s=4)
        axes[2].set_title('Boxplot of F1 scores', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('F1 score [%]', fontsize=12)
        axes[2].annotate(f'median: {f1_median:.2f} %', fontsize=9, color='red',
                        xy=(0.5, 1), xycoords='axes fraction', ha='left', verticalalignment='bottom',
                        xytext=(0.4, 0.9), textcoords='axes fraction')
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