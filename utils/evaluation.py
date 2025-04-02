import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

from config import logger

def evaluate_metrics(reference_answer, predicted_answer):
    """
    Calculate BLEU and ROUGE scores for a given reference and predicted answer.
    
    Args:
        reference_answer: The ground truth answer
        predicted_answer: The model generated answer
        
    Returns:
        Dictionary with BLEU and ROUGE scores
    """
    scores = {}
    
    # Calculate BLEU with smoothing
    reference_tokens = reference_answer.lower().split()
    predicted_tokens = predicted_answer.lower().split()
    smoothing_function = SmoothingFunction().method2
    scores["bleu"] = sentence_bleu([reference_tokens], predicted_tokens, smoothing_function=smoothing_function)
    
    # Calculate ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_answer, predicted_answer)
    scores["rouge1"] = rouge_scores['rouge1'].fmeasure
    scores["rouge2"] = rouge_scores['rouge2'].fmeasure
    scores["rougeL"] = rouge_scores['rougeL'].fmeasure
    
    return scores