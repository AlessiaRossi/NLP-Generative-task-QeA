import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from config import logger

def tune_bm25_params(qa_system, split="test", k=5, sample_size=50, 
                    k1_values=[1.2, 1.4, 1.6], b_values=[0.5, 0.75, 0.9]):
    """
    Perform a mini-grid search on BM25 parameters k1 and b to optimize retrieval metrics.
    
    Args:
        qa_system: The BioAsqQASystem instance
        split: Dataset split to use for evaluation
        k: Number of passages to retrieve
        sample_size: Number of questions to sample for evaluation
        k1_values: List of k1 parameters to try
        b_values: List of b parameters to try
    
    Returns:
        best_k1, best_b: The best performing parameters
    """
    if 'corpus' not in qa_system.dataset:
        logger.warning("No corpus found, cannot tune BM25 parameters.")
        return None, None

    # Prepare passages for BM25
    corpus = list(qa_system.passage_dict.values())
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    passage_keys = list(qa_system.passage_dict.keys())

    best_score = 0.0
    best_params = (None, None)

    logger.info("Starting BM25 parameter grid search...")
    for k1 in k1_values:
        for b_ in b_values:
            # Initialize BM25 with current k1, b parameters
            bm25_temp = BM25Okapi(tokenized_corpus, k1=k1, b=b_)

            # Temporarily set up for evaluation
            qa_system.bm25_index = bm25_temp
            qa_system.bm25_mapping = passage_keys
            qa_system._last_passage_count = len(qa_system.passage_dict)

            # Evaluate with current parameters
            metrics = qa_system.evaluate_retrieval(split=split, k=k, sample_size=sample_size, use_hybrid=True)

            recall = metrics.get("recall", 0)
            logger.info(f"BM25 config (k1={k1}, b={b_}): recall@{k}={recall:.4f}")

            # Update if better recall found
            if recall > best_score:
                best_score = recall
                best_params = (k1, b_)

    logger.info(f"Grid search completed. Best recall={best_score:.4f} with k1={best_params[0]}, b={best_params[1]}.")

    # Set BM25 with best parameters
    qa_system.bm25_index = BM25Okapi(tokenized_corpus, k1=best_params[0], b=best_params[1])
    qa_system.bm25_mapping = passage_keys
    qa_system._last_passage_count = len(qa_system.passage_dict)
    
    return best_params[0], best_params[1]