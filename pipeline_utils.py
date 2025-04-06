import os
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from evaluation_utils import *


def read_dataset(corpus_id):
    """
    Loads a dataset based on the given corpus ID and returns the relevant information for further processing.

    This function loads the content of a local markdown file corresponding to the specified corpus ID, 
    and a CSV file ('questions_df.csv'), both of which should be located in the 'dataset' folder.
    These files can be downloaded from the following resources:
    - https://github.com/brandonstarxel/chunking_evaluation/tree/main/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora
    - https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/evaluation_framework/general_evaluation_data/questions_df.csv

    Parameters:
    ----------
    corpus_id (str): The ID of the corpus to be loaded. This ID determines which dataset is read.

    Returns:
    ----------
    tuple: A tuple containing:
        - corpora (str): The content of the markdown file corresponding to the given corpus ID.
        - queries (pandas.Series): A series containing the queries associated with the corpus.
        - relevant_excerpts (pandas.Series): A series of dictionaries with the following keys:
            - 'content': The excerpt text.
            - 'start_index': The starting index of the excerpt in the document.
            - 'end_index': The ending index of the excerpt in the document.
    """

    md_file = os.path.join('dataset',corpus_id+'.md')

    with open(md_file, "r", encoding="utf-8") as file:
        corpora = file.read()

    questions_df = pd.read_csv(os.path.join('dataset','questions_df.csv'))

    relevant_questions_df = questions_df[questions_df.iloc[:,-1] == corpus_id]

    queries = relevant_questions_df.iloc[:,0]

    relevant_excerpts = relevant_questions_df.iloc[:,1]

    relevant_excerpts = relevant_excerpts.map(lambda x: json.loads(x))  # dict keys: "content", "start_index", "end_index"

    return corpora, queries, relevant_excerpts


def chunking_function(corpora, chunker):
    """
    Splits the given text (corpora) into chunks and generates metadata for each chunk.

    This function uses a chunker object to divide the provided text into smaller segments (chunks).
    For each chunk, it determines its starting and ending indices within the original document, 
    and stores this information as metadata.

    Parameters:
    ----------
    corpora (str): The input text to be split into chunks.
    chunker (object): A chunker object that implements the 'split_text' method to divide the text into chunks. 
                      An example of such a chunker is 'FixedTokenChunker' from:
                      https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/fixed_token_chunker.py

    Returns:
    ----------
    tuple: A tuple containing:
        - chunks (list): A list of text chunks obtained by splitting the input 'corpora'.
        - chunk_metadata (list): A list of dictionaries, each containing metadata for a chunk with the following keys:
            - 'start_index': The index where the chunk starts in the original 'corpora'.
            - 'end_index': The index where the chunk ends in the original 'corpora'.
    """

    chunks = chunker.split_text(corpora)
    chunk_metadata = []

    for chunk in chunks:

        start_ind, end_ind = find_target_in_document(corpora, chunk)

        chunk_metadata.append({"start_index": start_ind, "end_index": end_ind})

    return chunks, chunk_metadata


def retrieval_function(queries_emb, chunks_emb, Nr):
    """
    Retrieves the top-N relevant chunks for each query based on cosine similarity.

    This function calculates the cosine similarity between the embeddings of the queries and the chunks, 
    and retrieves the top-N most similar chunks for each query. The results are returned as both the 
    indices of the top-N relevant chunks and their corresponding cosine similarity scores.

    Parameters:
    ----------
    queries_emb (numpy.ndarray or list): A 2D array or list of query embeddings, where each row represents an embedding for a query.
    chunks_emb (numpy.ndarray or list): A 2D array or list of chunk embeddings, where each row represents an embedding for a chunk.
    Nr (int): The number of top relevant chunks to retrieve for each query based on cosine similarity.

    Returns:
    ----------
    tuple: A tuple containing:
        - top_ids (numpy.ndarray): A 2D array of shape (Nq, Nr) where Nq is the number of queries, and 
                                   Nr is the number of top chunks to retrieve. Each row contains the 
                                   indices of the top-N relevant chunks for the corresponding query.
        - cos_scores (numpy.ndarray): A 2D array of shape (Nq, Nr) where each row contains the cosine 
                                      similarity scores corresponding to the top-N chunks for the query.

    """

    Nq = len(queries_emb)

    top_ids = np.zeros((Nq,Nr))
    cos_scores = np.zeros((Nq,Nr))

    for i in range(Nq):

        cos_scores_tmp = util.pytorch_cos_sim(queries_emb[i], chunks_emb)[0].numpy()
        top_ids_tmp = np.argsort(cos_scores_tmp, axis=0)[::-1][:Nr]
        
        top_ids[i,:] = top_ids_tmp
        cos_scores[i,:] = cos_scores_tmp[top_ids_tmp]

    top_ids = top_ids.astype(int)

    return top_ids, cos_scores
