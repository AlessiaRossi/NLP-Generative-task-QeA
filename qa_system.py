import logging
import os
import json
import torch
import time
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import faiss

from config import logger, device, MODEL_NAME, ENCODER_NAME, TRANSFORMER_NAME
from utils.dataset_manager import load_bioasq_dataset, explore_dataset, preprocess_data, build_passage_index
from utils.retrieval import normalize_id, expand_biomedical_query, parse_relevant_passage_ids
from utils.evaluation import evaluate_metrics
from utils.setup_utils import setup_baseline, save_system_components, load_system_components
from utils.model_training import fine_tune_model, apply_peft
from utils.grid_search import tune_bm25_params
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

class BioAsqQASystem:
    def __init__(self, use_fine_tuning=True, use_peft=False, use_rag=True, models_dir="./models"):
        """
        Initialize the BioAsq Q&A System.
        
        Args:
            use_fine_tuning: Whether to fine-tune a small model
            use_peft: Whether to use Parameter-Efficient Fine-Tuning
            use_rag: Whether to use Retrieval-Augmented Generation
            models_dir: Directory to save/load models and related components
        """
        self.use_fine_tuning = use_fine_tuning
        self.use_peft = use_peft
        self.use_rag = use_rag
        self.models_dir = models_dir
        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self.faiss_index = None
        self.passage_dict = {}
        self.passage_embeddings = None
        self.indexed_passage_ids = None
        self.bm25_index = None
        self.bm25_mapping = None
        self._last_passage_count = 0
        self.debug = False
        
        # Initialize retrieval components if needed
        if use_rag:
            self.reranker = CrossEncoder(ENCODER_NAME)
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def load_dataset(self):
        """Load the BioAsq dataset from Hugging Face"""
        self.dataset = load_bioasq_dataset()
    
    def explore_dataset(self):
        """Print information about the dataset structure"""
        explore_dataset(self.dataset)
    
    def preprocess_data(self):
        """Preprocess the dataset for training and evaluation"""
        self.passage_dict = preprocess_data(self.dataset, self.models_dir)
        
        # For RAG, build passage index
        if self.use_rag:
            self._build_passage_index()
    
    def _normalize_id(self, pid):
        """Convert ID to integer if it's a numeric string"""
        return normalize_id(pid)
    
    def _build_passage_index(self):
        """Build an index for passage retrieval using FAISS"""
        self.faiss_index, self.indexed_passage_ids, self.passage_embeddings, self.retriever = build_passage_index(
            self.passage_dict, self.models_dir)
    
    def setup_baseline(self, model_name=MODEL_NAME):
        """Set up a baseline model"""
        setup_baseline(self, model_name)
    
    def fine_tune(self, output_dir="./fine_tuned_model", num_train_epochs=3, force_train=False):
        """Fine-tune the model on the BioAsq dataset"""
        fine_tune_model(self, output_dir, num_train_epochs, force_train)
    
    def apply_peft(self, output_dir="./peft_model", num_train_epochs=3, force_train=False):
        """Apply Parameter-Efficient Fine-Tuning using LoRA"""
        apply_peft(self, output_dir, num_train_epochs, force_train)
    
    def save_all_components(self, output_dir=None):
        """Save all model components and metadata"""
        return save_system_components(self, output_dir)
    
    def load_all_components(self, input_dir):
        """Load all model components and metadata from a saved directory"""
        return load_system_components(self, input_dir)
    
    def _retrieve_passages(self, question, k=5):
        """Retrieve k most relevant passages for a given question"""
        if not self.use_rag or not hasattr(self, 'faiss_index') or self.faiss_index is None:
            return []
        
        # Encode the question
        question_embedding = self.retriever.encode(question, convert_to_tensor=True, show_progress_bar=False)
        question_embedding_np = question_embedding.cpu().numpy().reshape(1, -1)
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(question_embedding_np, k)
        
        # Get retrieved passages
        retrieved_passages = []
        for idx in indices[0]:
            passage_id = self.indexed_passage_ids[idx]
            passage_id = self._normalize_id(passage_id)
            
            if passage_id in self.passage_dict:
                passage_text = self.passage_dict[passage_id]
                retrieved_passages.append({
                    "id": passage_id,
                    "text": passage_text
                })
                
        return retrieved_passages
    
    def rerank_passages(self, question, retrieved_passages, top_k=5):
        """Rerank passages using a cross-encoder model for more accurate relevance scoring
        Args:
            question: The input question
            retrieved_passages: List of retrieved passages to rerank
            top_k: Number of top passages to return after reranking

        Returns:
            List of reranked passages
        """
        # First check if reranker exists, initialize if needed
        if not hasattr(self, 'reranker'):
            self.reranker = CrossEncoder(ENCODER_NAME)

        # Optimize input format for biomedical context
        pairs = [(f"[BIOMEDICAL QUESTION] {question}", f"[BIOMEDICAL PASSAGE] {p['text']}") for p in retrieved_passages]
        
        # Get scores from reranker
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        # Apply bonus for biomedical term matches
        medical_terms = ["gene", "protein", "receptor", "pathway", "enzyme", "cell", "tissue", "disease", "syndrome"]
        for i, passage in enumerate(retrieved_passages):
            for term in medical_terms:
                if term in question.lower() and term in passage["text"].lower():
                    scores[i] += 0.2
        
        # Combine passages with their scores
        reranked = [(retrieved_passages[i], scores[i]) for i in range(len(scores))]
        
        # Sort and return top_k passages
        return [item[0] for item in sorted(reranked, key=lambda x: x[1], reverse=True)[:top_k]]
    
    def expand_query(self, question):
        """Expand query with biomedical synonyms to improve retrieval performance"""
        return expand_biomedical_query(question)
    
    def generate_answer(self, question, use_rag=True, use_reranking=True, use_hybrid=True):
        """Generate an answer to a given question with optional reranking and hybrid retrieval

        Args:
            question: The input question
            use_rag: Whether to use retrieval-augmented generation
            use_reranking: Whether to apply reranking on retrieved passages
            use_hybrid: Whether to use hybrid retrieval (semantic + BM25)

        Returns:
            Generated answer as a string
        """
        context = ""
        if use_rag and self.use_rag:
            # Retrieve passages using either hybrid retrieval or standard retrieval
            if use_hybrid:
                # Use hybrid retrieval (combining semantic and BM25)
                initial_passages = self.hybrid_retrieve(question, k=10)
            else:
                # Use only semantic retrieval
                initial_passages = self._retrieve_passages(question, k=10)
            
            # Apply reranking if enabled
            if use_reranking and hasattr(self, 'reranker'):
                retrieved_passages = self.rerank_passages(question, initial_passages, top_k=5)
            else:
                retrieved_passages = initial_passages[:5]
            
            # Combine passage texts
            context = " ".join([p["text"] for p in retrieved_passages])
        
        # Format input
        input_text = f"question: {question}"
        if context:
            input_text += f" context: {context}"
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate answer
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        
        # Decode output
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def evaluate(self, split="test", sample_size=100):
        """Evaluate the model on the test set, using a random sample of questions for efficiency

        Args:
            split: The dataset split to evaluate on (e.g., "train", "test", "dev")
            sample_size: Number of questions to sample for evaluation

        Returns:
            Dictionary with evaluation metrics (e.g., BLEU, ROUGE)
        """
        logger.info(f"Evaluating model on {split} set (sampling {sample_size} questions)...")
        
        # Make sure the requested split exists
        if split not in self.dataset:
            available_splits = [s for s in self.dataset.keys() if s != "corpus"]
            if not available_splits:
                logger.error("No evaluation splits available")
                return {}
            split = available_splits[0]
            logger.warning(f"Requested split '{split}' not found. Using '{split}' instead.")
        
        # Get full eval dataset
        eval_data = self.dataset[split]
        # Sample a subset for evaluation to save time
        total_examples = len(eval_data)
        if sample_size >= total_examples:
            sampled_indices = list(range(total_examples))
        else:
            sampled_indices = list(range(total_examples))
            np.random.shuffle(sampled_indices)
            sampled_indices = sampled_indices[:sample_size]
        
        logger.info(f"Evaluating on {len(sampled_indices)} examples from {total_examples} total examples")
        
        results = {
            "bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }
        
        for idx in tqdm(sampled_indices, desc="Evaluating", unit="question"):
            item = eval_data[idx]
            question = item["question"]
            reference_answer = item["answer"]
            
            # Generate prediction
            predicted_answer = self.generate_answer(question)
            
            # Calculate metrics
            scores = evaluate_metrics(reference_answer, predicted_answer)
            
            # Collect scores
            for metric, score in scores.items():
                results[metric].append(score)
        
        # Calculate average scores
        avg_results = {metric: np.mean(scores) for metric, scores in results.items()}
        
        logger.info(f"Evaluation results: {avg_results}")
        print(f"DEBUG - Evaluation results: {avg_results}")
        return avg_results
    
    def evaluate_retrieval(self, split="test", k=5, sample_size=100, use_hybrid=True):
        """Evaluate the retrieval component using Recall@k and Precision@k on a random subset

        Args:
            split: The dataset split to evaluate on (e.g., "train", "test", "dev")
            k: Number of passages to retrieve
            sample_size: Number of questions to sample for evaluation
            use_hybrid: Whether to use hybrid retrieval (semantic + BM25)

        Returns:
        - A dictionary with average recall and precision metrics.
        """
        if not self.use_rag:
            logger.info("RAG is disabled. Skipping retrieval evaluation.")
            return {}
            
        logger.info(f"Evaluating retrieval on {split} set with k={k} (sampling {sample_size} questions)...")
        
        # Make sure the requested split exists
        if split not in self.dataset:
            available_splits = [s for s in self.dataset.keys() if s != "corpus"]
            if not available_splits:
                logger.error("No evaluation splits available")
                return {}
            split = available_splits[0]
            logger.warning(f"Requested split '{split}' not found. Using '{split}' instead.")
        
        # Get full eval dataset
        eval_data = self.dataset[split]
        # Sample a subset for evaluation to save time
        total_examples = len(eval_data)
        sampled_indices = list(range(total_examples))
        if sample_size < total_examples:
            np.random.shuffle(sampled_indices)
            sampled_indices = sampled_indices[:sample_size]
        
        logger.info(f"Evaluating retrieval on {len(sampled_indices)} examples from {total_examples} total examples")
        
        recalls = []
        precisions = []

        for idx in tqdm(sampled_indices, desc="Evaluating", unit="question"):
            item = eval_data[idx]
            question = item["question"]
            relevant_passage_ids = parse_relevant_passage_ids(item.get("relevant_passage_ids", "[]"))
            
            # Skip if no relevant passages
            if not relevant_passage_ids:
                continue
                
            # Get retrieved passage IDs using either hybrid or standard retrieval
            if use_hybrid:
                retrieved_passages = self.hybrid_retrieve(question, k=k)
            else:
                retrieved_passages = self._retrieve_passages(question, k=k)
            retrieved_ids = [p["id"] for p in retrieved_passages]
            
            # Calculate metrics
            relevant_retrieved = set(relevant_passage_ids).intersection(set([self._normalize_id(pid) for pid in retrieved_ids]))
            recall = len(relevant_retrieved) / len(relevant_passage_ids) if relevant_passage_ids else 0
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            
            recalls.append(recall)
            precisions.append(precision)
        
        avg_recall = np.mean(recalls)
        avg_precision = np.mean(precisions)
        
        logger.info(f"Retrieval evaluation: Recall@{k}={avg_recall:.4f}, Precision@{k}={avg_precision:.4f}")
        print(f"DEBUG - Retrieval evaluation: Recall@{k}: {avg_recall:.4f}, Precision@{k}: {avg_precision:.4f}")
        return {"recall": avg_recall, "precision": avg_precision}

    def hybrid_retrieve(self, question, k=10):
        """Combines semantic and BM25 search with improved retrieval strategy

        Args:
            question: The input question
            k: Number of passages to retrieve

        Returns:
            List of retrieved passages
        """
        # Initialize BM25 if needed
        if not hasattr(self, 'bm25_index') or not hasattr(self, 'bm25_mapping') \
        or not hasattr(self, '_last_passage_count') or self._last_passage_count != len(self.passage_dict):
            logger.info("Initializing or updating BM25 index")
            # Prepare data for BM25
            corpus = list(self.passage_dict.values())
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus, k1=1.6, b=0.75)
            self.bm25_mapping = list(self.passage_dict.keys())
            self._last_passage_count = len(self.passage_dict)

        # 1. Expanded query for better term matching
        expanded_question = self.expand_query(question)

        # 2. Get semantic search results (increase candidate pool)
        semantic_passages = self._retrieve_passages(question, k=k*2)
        semantic_ids = [p["id"] for p in semantic_passages]
        
        # 3. Get BM25 results using the expanded query
        tokenized_query = expanded_question.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_n = min(k*2, len(bm25_scores))
        bm25_indices = np.argsort(bm25_scores)[-top_n:][::-1]
        bm25_ids = [self._normalize_id(self.bm25_mapping[idx]) for idx in bm25_indices]

        # 4. Combine with interleaving for better diversity
        combined_passages = []
        seen_ids = set()

        # Interleave results from both methods
        for i in range(max(len(semantic_ids), len(bm25_ids))):
            # Add semantic result if available and not already added
            if i < len(semantic_ids) and semantic_ids[i] not in seen_ids:
                pid = semantic_ids[i]
                if pid in self.passage_dict:
                    combined_passages.append({
                        "id": pid, 
                        "text": self.passage_dict[pid],
                        "source": "semantic"
                    })
                    seen_ids.add(pid)
            
            # Add BM25 result if available and not already added
            if i < len(bm25_ids) and bm25_ids[i] not in seen_ids:
                pid = bm25_ids[i]
                if pid in self.passage_dict:
                    combined_passages.append({
                        "id": pid,
                        "text": self.passage_dict[pid],
                        "source": "bm25"
                    })
                    seen_ids.add(pid)

        # 5. Optional: Re-rank the combined results for final ordering
        if hasattr(self, 'reranker'):
            return self.rerank_passages(question, combined_passages[:k*2], top_k=k)

        return combined_passages[:k]
    
    def tune_bm25_params(self, split="test", k=5, sample_size=50,
                      k1_values=[1.2, 1.4, 1.6], b_values=[0.5, 0.75, 0.9]):
        """
        Perform a mini-grid search on BM25 parameters k1 and b to optimize retrieval metrics.
        """
        return tune_bm25_params(self, split, k, sample_size, k1_values, b_values)

    def set_debug_mode(self, debug=True):
        """Enable or disable debug mode for detailed logging"""
        self.debug = debug
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        if debug:
            logger.info("Debug mode enabled")
        else:
            logger.info("Debug mode disabled")