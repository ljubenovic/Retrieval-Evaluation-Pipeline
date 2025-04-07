from pipeline_utils import read_dataset
import tiktoken


def analyze_relevant_excerpts(corpus_id):
    """
    Analyzes the relevant excerpts associated with a given corpus and given queries by computing:
    1. The number of highlighted excerpts for each query.
    2. The number of tokens (based on 'cl100k_base' encoding) per each highlighted excerpt.

    Parameters:
    ----------
    corpus_id (str): Identifier of the corpus to be loaded using 'read_dataset'.

    Returns:
    ----------
    highlights_per_query (list of int): A list containing the number of highlights associated with each query.
    tokens_per_highlight (list of int): A list containing the token count for each highlight across all queries,
                                        based on the 'cl100k_base' tokenizer.
    """

    corpus, queries, relevant_excerpts = read_dataset(corpus_id)

    encoding = tiktoken.get_encoding('cl100k_base')

    N_queries = len(queries)
    highlights_per_query = []
    tokens_per_highlight = []

    for query_index in range(N_queries):

        highlights = relevant_excerpts[query_index]
        highlights_per_query.append(len(highlights))

        tokens_per_highlight_per_query = []
        for highlight in highlights:
            highlight_tokens = encoding.encode(highlight['content'])
            
            N_tokens = len(highlight_tokens)
            tokens_per_highlight_per_query.append(N_tokens)
        
        tokens_per_highlight.extend(tokens_per_highlight_per_query)

    return highlights_per_query, tokens_per_highlight

