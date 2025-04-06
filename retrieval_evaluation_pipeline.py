from pipeline_utils import read_dataset, chunking_function, retrieval_function
from evaluation import calculate_metrics

def retrieval_evaluation_pipeline(corpus_id, chunker, embedding_function, N, show_plots=False):
    """
    Executes a full retrieval evaluation pipeline, including data loading, chunking, embedding, retrieval, and evaluation.

    This function loads a dataset based on the given 'corpus_id', splits the corpus into chunks using the provided 'chunker',
    generates embeddings for both the chunks and the queries using the specified 'embedding_function', retrieves the top-N
    most relevant chunks for each query based on cosine similarity, and evaluates the retrieval performance using standard
    information retrieval (IR) metrics such as precision, recall, and F1-score.

    Parameters:
    ----------
    corpus_id (str): Identifier for the corpus to be used. The function 'read_dataset' will use this ID to load the dataset
                    (including the corpus, associated queries, and ground-truth relevant excerpts).
    chunker (object): An object implementing a 'split_text' method, used to divide the corpus into smaller chunks (e.g., 'FixedTokenChunker').
    embedding_function (Callable): A function that takes a string and returns its embedding (vector representation) via a pre-trained sentence transformer model.
    N (int): Number of top retrieved chunks per query to return. This controls the retrieval depth.
    show_plots (bool, optional): Whether or not to display boxplots of the precision, recall, and F1-score metrics. Default is False.

    Returns:
    ----------
    metrics (pandas.DataFrame): A DataFrame containing the precision, recall, and F1 score for each query.
    metrics_summary (pandas.DataFrame): A summary DataFrame with the mean and standard deviation of precision, recall, and F1 score across all queries.
    """

    # Data loading
    corpora, queries, relevant_excerpts = read_dataset(corpus_id)
    
    # Corpora chunking
    chunks, chunk_metadata = chunking_function(corpora, chunker)
    
    # Embedding
    chunks_emb = [embedding_function(chunk) for chunk in chunks]
    queries_emb = [embedding_function(query) for query in queries]
    
    # Retrieval
    retrieved_ids, _ = retrieval_function(queries_emb, chunks_emb, N)
    
    # Evaluation
    metrics, metrics_summary, highlighted_chunks_count = calculate_metrics(relevant_excerpts, retrieved_ids, chunk_metadata, show_plots)
    
    return metrics, metrics_summary