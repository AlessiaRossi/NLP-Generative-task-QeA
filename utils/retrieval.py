import numpy as np
import json
from rank_bm25 import BM25Okapi

from config import logger

def create_bm25_index(corpus_texts, k1=1.6, b=0.75):
    """
    Create a BM25 index for a corpus of texts
    
    Args:
        corpus_texts: List of text passages
        k1, b: BM25 parameters
        
    Returns:
        BM25 index and tokenized corpus
    """
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    return bm25_index, tokenized_corpus

def normalize_id(pid):
    """Convert ID to integer if it's a numeric string"""
    if isinstance(pid, str) and pid.isdigit():
        return int(pid)
    return pid

def expand_biomedical_query(question):
    """Expand query with biomedical synonyms to improve retrieval performance

    Args:
        question (str): The input question string.

    Returns:
        str: The expanded question string with synonyms.
    """
    # Comprehensive biomedical term dictionary
    medical_terms = {
        # General terms
        "cancer": ["neoplasm", "tumor", "malignancy", "carcinoma", "oncology"],
        "heart attack": ["myocardial infarction", "cardiac arrest", "coronary thrombosis"],
        "stroke": ["cerebrovascular accident", "brain attack", "cerebral infarction"],
        "diabetes": ["diabetes mellitus", "hyperglycemia", "insulin resistance"],
        "high blood pressure": ["hypertension", "elevated blood pressure"],
        
        # Genetic terms
        "gene": ["allele", "locus", "genetic marker", "DNA sequence"],
        "protein": ["polypeptide", "amino acid chain", "enzyme", "receptor"],
        "mutation": ["genetic variation", "polymorphism", "alteration", "variant"],
        "chromosome": ["chromatin", "karyotype", "genome", "DNA"],
        "dna": ["genetic material", "nucleic acid", "genome", "double helix"],
        
        # Disease terms
        "disease": ["disorder", "syndrome", "pathology", "condition", "illness"],
        "infection": ["infectious disease", "pathogen", "microbial invasion"],
        "inflammation": ["inflammatory response", "swelling", "edema"],
        "syndrome": ["disorder", "condition", "disease complex", "symptom group"],
        "autoimmune": ["immune disorder", "self-antibodies", "autoimmunity"],
        
        # Cell/molecular terms
        "cell": ["cellular", "cytoplasm", "organelle"],
        "pathway": ["signaling cascade", "biochemical route", "molecular mechanism"],
        "receptor": ["binding site", "recognition site", "cell surface protein"],
        "enzyme": ["catalyst", "biological catalyst", "protein catalyst"],
        "antibody": ["immunoglobulin", "immune protein", "antibodies"]
    }
    
    # Extract all words from question
    words = question.lower().split()
    expanded = question
    
    # Check multi-word terms first
    for term, synonyms in medical_terms.items():
        if " " in term and term.lower() in question.lower():
            expanded += " " + " ".join(synonyms)
    
    # Then check single words
    for word in words:
        for term, synonyms in medical_terms.items():
            if " " not in term and word == term.lower():
                expanded += " " + " ".join(synonyms)
                
    return expanded

def parse_relevant_passage_ids(passage_ids_str):
    """Parse the string representation of passage IDs into a list"""
    try:
        relevant_passage_ids = json.loads(passage_ids_str.replace("'", '"'))
        return [normalize_id(pid) for pid in relevant_passage_ids]
    except (json.JSONDecodeError, AttributeError):
        logger.warning(f"Could not parse passage IDs: {passage_ids_str}")
        return []