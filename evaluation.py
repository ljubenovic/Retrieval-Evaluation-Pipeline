import pandas as pd
from evaluation_utils import *
from visualization_utils import plot_metrics_boxplots


def calculate_metrics(relevant_excerpts, retrieved_ids, chunk_metadata, show_plots=True):
    """
    Calculates evaluation metrics (precision, recall, and F1 score) for the retrieved chunks based on the relevant excerpts.

    This function computes the precision, recall, and F1 score for each query.
    Additionally, it calculates the number of highlighted chunks for each query.
    The metrics are returned as both individual values for each query and summary statistics (mean and standard deviation) across all queries.
    Optionally, it can display boxplots of the metrics.

    Parameters:
    ----------
    relevant_excerpts (list of list of dicts): A list of relevant excerpts for each query. Each excerpt is a dictionary with 'start_index' and 'end_index' keys 
                                            indicating the range of the excerpt in the original document.
    retrieved_ids (numpy.ndarray): A 2D array where each row represents a query and each column contains the ID of a retrieved chunk.
                                   The IDs correspond to the index in 'chunk_metadata'.
    chunk_metadata (list of dicts): A list of dictionaries containing metadata for each chunk, including 'start_index' and 'end_index'.
    show_plots (bool, optional): If True, displays boxplots of the precision, recall, and F1 scores. Default is True.

    Returns:
    ----------
    tuple: A tuple containing:
        - metrics (pandas.DataFrame): A DataFrame with precision, recall, and F1 score for each query.
        - metrics_summary (pandas.DataFrame): A DataFrame with the mean and standard deviation of precision, recall, and F1 score across all queries.
        - highlighted_chunks_count (list): A list of the number of highlighted chunks for each query.
    """

    N_queries, _ = retrieved_ids.shape

    precision_scores = []
    recall_scores = []
    f1_scores = []

    highlighted_chunks_count = []

    for query_index in range(N_queries):

        references = relevant_excerpts[query_index]

        used_highlights = []
        highlighted_chunk_count = 0

        for id in retrieved_ids[query_index,:]:

            chunk_start = chunk_metadata[id]["start_index"]
            chunk_end = chunk_metadata[id]["end_index"]

            contains_highlight = False

            for reference in references:

                ref_start = int(reference["start_index"])
                ref_end = int(reference["end_index"])

                intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
    
                if intersection is not None:
                    contains_highlight = True
                    used_highlights = union_ranges([*used_highlights, intersection])

            if contains_highlight:
                    highlighted_chunk_count += 1

        highlighted_chunks_count.append(highlighted_chunk_count)

        precision = sum_of_ranges(used_highlights)/sum_of_ranges([(chunk_metadata[id]["start_index"], chunk_metadata[id]["end_index"]) for id in retrieved_ids[query_index,:]])
        recall = sum_of_ranges(used_highlights)/sum_of_ranges([(int(ref["start_index"]), int(ref["end_index"])) for ref in references])
        f1 = 2*precision*recall/(precision+recall) if (precision or recall) else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    metrics = pd.DataFrame({
        'precision': precision_scores,
        'recall': recall_scores,
        'f1_score': f1_scores
    })

    metrics_summary = pd.DataFrame({
        'precision_mean': [metrics['precision'].mean()*100],
        'precision_std': [metrics['precision'].std()*100],
        'recall_mean': [metrics['recall'].mean()*100],
        'recall_std': [metrics['recall'].std()*100],
        'f1_mean': [metrics['f1_score'].mean()*100],
        'f1_std': [metrics['f1_score'].std()*100]
    })

    print('Evaluation results:')
    print('\tPrecision: {:.2f} ± {:.2f} %'.format(metrics_summary['precision_mean'].values[0], metrics_summary['precision_std'].values[0]))
    print('\tRecall: {:.2f} ± {:.2f} %'.format(metrics_summary['recall_mean'].values[0], metrics_summary['recall_std'].values[0]))
    print('\tF1 score: {:.2f} ± {:.2f} %'.format(metrics_summary['f1_mean'].values[0], metrics_summary['f1_std'].values[0]))

    if show_plots:
        plot_metrics_boxplots(metrics)

    return metrics, metrics_summary, highlighted_chunks_count