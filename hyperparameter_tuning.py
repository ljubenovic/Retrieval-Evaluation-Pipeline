
import itertools
import pandas as pd
import datetime
from retrieval_evaluation_pipeline import *

def grid_search(corpus_id, chunker, embedding_function, chunk_size_values, overlap_percentages, Nr_values):
    
    results = {}
    results_str = pd.DataFrame(columns=['chunk_size', 'chunk_overlap', 'Nr', 'precision', 'recall', 'f1'])

    param_combinations = itertools.product(chunk_size_values, overlap_percentages, Nr_values)

    for chunk_size, overlap_percentage, Nr in param_combinations:

        chunk_overlap = int(overlap_percentage*chunk_size/100)
        print(f"Testing chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, Nr={Nr}...")
        
        chunker.chunk_size = chunk_size
        chunker.chunk_overlap = chunk_overlap
        
        metrics, metrics_summary = retrieval_evaluation_pipeline(corpus_id, chunker, embedding_function, Nr, show_plots=False)
        
        results[(chunk_size, chunk_overlap, Nr)] = {
            'precision_mean': metrics_summary['precision_mean'].item(),
            'precision_std': metrics_summary['precision_std'].item(),
            'recall_mean': metrics_summary['recall_mean'].item(),
            'recall_std': metrics_summary['recall_std'].item(),
            'f1_mean': metrics_summary['f1_mean'].item(),
            'f1_std': metrics_summary['f1_std'].item()
        }

        row = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'Nr': Nr,
            'precision': f"{metrics_summary['precision_mean'].item():.2f} ± {metrics_summary['precision_std'].item():.2f}",
            'recall': f"{metrics_summary['recall_mean'].item():.2f} ± {metrics_summary['recall_std'].item():.2f}",
            'f1': f"{metrics_summary['f1_mean'].item():.2f} ± {metrics_summary['f1_std'].item():.2f}"
        }
        results_str = pd.concat([results_str, pd.DataFrame([row])], ignore_index=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_str.to_csv(f"results_{timestamp}.csv", index=False)
    
    return results, results_str

