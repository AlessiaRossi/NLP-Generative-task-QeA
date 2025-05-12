import os
import torch
import json
import pickle
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from config import (
    logger, device, DATASET_NAME, DATASET_SUBSET, 
    TRANSFORMER_NAME
)
from utils.retrieval import normalize_id

def load_bioasq_dataset():
    """Load the BioAsq dataset from Hugging Face with proper configuration"""
    logger.info("Loading BioAsq dataset...")
    
    # Load both configurations of the dataset
    corpus = load_dataset(DATASET_NAME, DATASET_SUBSET[1])
    qa_passages = load_dataset(DATASET_NAME, DATASET_SUBSET[0])
    
    # Initialize dataset dictionary
    dataset = {}
    
    # Store corpus data - check what splits are available instead of assuming 'train'
    logger.info(f"Corpus splits available: {list(corpus.keys())}")
    corpus_split = list(corpus.keys())[0]  # Use the first available split
    dataset["corpus"] = corpus[corpus_split]
    
    # Store QA data
    logger.info(f"QA passages splits available: {list(qa_passages.keys())}")
    
    # Map available splits from qa_passages
    for split in qa_passages.keys():
        dataset[split] = qa_passages[split]
    
    # Log dataset info
    logger.info(f"Dataset loaded with {len(dataset['corpus'])} corpus documents")
    for split in qa_passages.keys():
        if split in dataset:
            logger.info(f"  {split} split: {len(dataset[split])} QA pairs")
    
    return dataset

def explore_dataset(dataset):
    """Print information about the dataset structure"""
    logger.info("Exploring dataset structure:")
    for split in dataset.keys():
        logger.info(f"Split: {split}, Size: {len(dataset[split])}")
        logger.info(f"Features: {dataset[split].features}")
        logger.info(f"Sample: {dataset[split][0]}")
    
def preprocess_data(dataset, models_dir):
    """Preprocess the dataset for training and evaluation"""
    logger.info("Preprocessing data...")
    
    # Check if processed passage dict exists
    passage_dict_path = os.path.join(models_dir, "passage_dict.pkl")
    if os.path.exists(passage_dict_path):
        logger.info(f"Loading preprocessed passage dictionary from {passage_dict_path}")
        with open(passage_dict_path, 'rb') as f:
            passage_dict = pickle.load(f)
        logger.info(f"Loaded {len(passage_dict)} passages")
    else:
        # Create passage dictionary from the corpus with INTEGER keys
        passage_dict = {}
        if 'corpus' in dataset:
            for item in dataset['corpus']:
                # Always store as integer if possible
                pid = item['id']
                if isinstance(pid, str) and pid.isdigit():
                    pid = int(pid)
                passage_dict[pid] = item['passage']
        
        # Save the passage dictionary
        logger.info(f"Saving passage dictionary with {len(passage_dict)} passages to {passage_dict_path}")
        with open(passage_dict_path, 'wb') as f:
            pickle.dump(passage_dict, f)
    
    return passage_dict

def build_passage_index(passage_dict, models_dir):
    """Build an index for passage retrieval using FAISS"""
    # Define paths for saved components
    index_dir = os.path.join(models_dir, "faiss_index")
    faiss_index_path = os.path.join(index_dir, "passage_index.faiss")
    passage_ids_path = os.path.join(index_dir, "passage_ids.json")
    embeddings_path = os.path.join(index_dir, "passage_embeddings.pt")
    
    # Check if index directory exists
    os.makedirs(index_dir, exist_ok=True)
    
    # Check if we can load pre-built index
    if os.path.exists(faiss_index_path) and os.path.exists(passage_ids_path) and os.path.exists(embeddings_path):
        logger.info("Loading pre-built FAISS index and related components...")
        
        # Load FAISS index
        faiss_index = faiss.read_index(faiss_index_path)
        logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors")
        
        # Load passage IDs
        with open(passage_ids_path, 'r') as f:
            indexed_passage_ids = json.load(f)
            
        # Load embeddings if needed
        passage_embeddings = torch.load(embeddings_path, map_location=device, weights_only=True)
            
        # Load retriever model
        retriever = SentenceTransformer(TRANSFORMER_NAME)
        retriever.to(device)
        
        logger.info("FAISS index and components loaded successfully")
        
        return faiss_index, indexed_passage_ids, passage_embeddings, retriever
    
    logger.info("Building new FAISS index for passage retrieval...")
    
    # Initialize sentence transformer model for embeddings
    retriever = SentenceTransformer(TRANSFORMER_NAME)
    retriever.to(device)
    
    # Get all passages and their IDs
    passages = list(passage_dict.values())
    passage_ids = list(passage_dict.keys())
    
    # Create embeddings
    passage_embeddings = retriever.encode(
        passages, 
        convert_to_tensor=True,
        show_progress_bar=True
    )
    
    # Convert to numpy for FAISS
    embeddings_np = passage_embeddings.cpu().numpy()

    # L2 normalization
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    # Add small epsilon to avoid division by zero
    norms[norms == 0] = 1e-8
    embeddings_np = embeddings_np / norms
    
    # Create FAISS index
    embedding_size = embeddings_np.shape[1]
    faiss_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(embedding_size), embedding_size, 256)
    faiss_index.train(embeddings_np)
    faiss_index.add(embeddings_np)
    
    # Increase nprobe for more accurate retrieval
    faiss_index.nprobe = 32
    
    # Store passage IDs for lookup
    indexed_passage_ids = passage_ids
    
    # Save all components
    logger.info(f"Saving FAISS index to {faiss_index_path}")
    faiss.write_index(faiss_index, faiss_index_path)
    
    logger.info(f"Saving passage IDs to {passage_ids_path}")
    with open(passage_ids_path, 'w') as f:
        json.dump(indexed_passage_ids, f)
    
    logger.info(f"Saving passage embeddings to {embeddings_path}")
    torch.save(passage_embeddings, embeddings_path)
    
    logger.info(f"FAISS index built with {len(passages)} passages and saved")
    
    return faiss_index, indexed_passage_ids, passage_embeddings, retriever