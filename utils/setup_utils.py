import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import logger, device, MODEL_NAME
from utils.dataset_manager import load_bioasq_dataset, explore_dataset, preprocess_data, build_passage_index

def setup_baseline(qa_system, model_name=MODEL_NAME):
    """Set up a baseline model using prompting or a small pre-trained model.
    """
    logger.info(f"Setting up baseline model: {model_name}")
    
    # Define model paths
    model_dir = os.path.join(qa_system.models_dir, "baseline")
    
    # Check if model already exists
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        logger.info(f"Loading baseline model from {model_dir}")
        qa_system.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        qa_system.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        qa_system.model.to(device)
        logger.info("Baseline model loaded successfully")
    else:
        # Download and save the model
        logger.info(f"Downloading baseline model {model_name}")
        qa_system.tokenizer = AutoTokenizer.from_pretrained(model_name)
        qa_system.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        qa_system.model.to(device)
        
        # Save the model
        logger.info(f"Saving baseline model to {model_dir}")
        qa_system.tokenizer.save_pretrained(model_dir)
        qa_system.model.save_pretrained(model_dir)

def initialize_system_components(qa_system):
    """Initialize dataset, preprocess data and prepare model components"""
    # Load the dataset
    qa_system.dataset = load_bioasq_dataset()
    explore_dataset(qa_system.dataset)
    
    # Preprocess data and get passage dictionary
    qa_system.passage_dict = preprocess_data(qa_system.dataset, qa_system.models_dir)
    
    # Build passage index for RAG if enabled
    if qa_system.use_rag:
        qa_system.faiss_index, qa_system.indexed_passage_ids, qa_system.passage_embeddings, qa_system.retriever = build_passage_index(
            qa_system.passage_dict, qa_system.models_dir)
    
    # Set up baseline model
    setup_baseline(qa_system)

def save_system_components(qa_system, output_dir=None):
    """Save all model components and metadata

    Args:
        qa_system (BioAsqQASystem): The QA system instance.
        output_dir (str): Directory to save the components. If None, a new directory will be created.

    Returns:
        str: Path to the saved directory.
    """
    import json
    import time
    import faiss
    from datetime import datetime
    
    if output_dir is None:
        output_dir = os.path.join(qa_system.models_dir, f"qa_system_{int(time.time())}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving all system components to {output_dir}")
    
    # Save model and tokenizer
    if qa_system.model is not None and qa_system.tokenizer is not None:
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        qa_system.model.save_pretrained(model_dir)
        qa_system.tokenizer.save_pretrained(model_dir)
        logger.info(f"Model and tokenizer saved to {model_dir}")
    
    # Save FAISS index and related components
    if qa_system.use_rag and qa_system.faiss_index is not None:
        index_dir = os.path.join(output_dir, "faiss_index")
        os.makedirs(index_dir, exist_ok=True)
        
        faiss_index_path = os.path.join(index_dir, "passage_index.faiss")
        passage_ids_path = os.path.join(index_dir, "passage_ids.json")
        embeddings_path = os.path.join(index_dir, "passage_embeddings.pt")
        
        faiss.write_index(qa_system.faiss_index, faiss_index_path)
        
        with open(passage_ids_path, 'w') as f:
            json.dump(qa_system.indexed_passage_ids, f)
        
        if qa_system.passage_embeddings is not None:
            torch.save(qa_system.passage_embeddings, embeddings_path)
            
        logger.info(f"FAISS index and related components saved to {index_dir}")
    
    # Save passage dictionary
    if qa_system.passage_dict:
        passage_dict_path = os.path.join(output_dir, "passage_dict.pkl")
        with open(passage_dict_path, 'wb') as f:
            import pickle
            pickle.dump(qa_system.passage_dict, f)
        logger.info(f"Passage dictionary saved to {passage_dict_path}")
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    config = {
        "use_fine_tuning": qa_system.use_fine_tuning,
        "use_peft": qa_system.use_peft,
        "use_rag": qa_system.use_rag,
        "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f)
        
    logger.info(f"Configuration saved to {config_path}")
    return output_dir

def load_system_components(qa_system, input_dir):
    """Load all model components and metadata from a saved directory"""
    import json
    import pickle
    import faiss
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"Loading system components from {input_dir}")
    
    # Load configuration
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        qa_system.use_fine_tuning = config.get("use_fine_tuning", qa_system.use_fine_tuning)
        qa_system.use_peft = config.get("use_peft", qa_system.use_peft)
        qa_system.use_rag = config.get("use_rag", qa_system.use_rag)
        
        logger.info(f"Loaded configuration: {config}")
    
    # Load model and tokenizer
    model_dir = os.path.join(input_dir, "model")
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        if qa_system.use_peft:
            # For PEFT loading we need a base model first
            base_model_path = os.path.join(qa_system.models_dir, "baseline")
            if not os.path.exists(base_model_path):
                # If base model doesn't exist, get it first
                setup_baseline(qa_system)
                
            qa_system.model = PeftModel.from_pretrained(
                qa_system.model,  # Base model
                model_dir,   # PEFT adapter weights
            )
        else:
            qa_system.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            
        qa_system.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        qa_system.model.to(device)
        logger.info(f"Model and tokenizer loaded from {model_dir}")
    
    # Load passage dictionary
    passage_dict_path = os.path.join(input_dir, "passage_dict.pkl")
    if os.path.exists(passage_dict_path):
        with open(passage_dict_path, 'rb') as f:
            qa_system.passage_dict = pickle.load(f)
        logger.info(f"Loaded {len(qa_system.passage_dict)} passages from dictionary")
    
    # Load FAISS index and related components
    if qa_system.use_rag:
        from config import TRANSFORMER_NAME
        index_dir = os.path.join(input_dir, "faiss_index")
        faiss_index_path = os.path.join(index_dir, "passage_index.faiss")
        passage_ids_path = os.path.join(index_dir, "passage_ids.json")
        embeddings_path = os.path.join(index_dir, "passage_embeddings.pt")
        
        if os.path.exists(faiss_index_path) and os.path.exists(passage_ids_path):
            # Load FAISS index
            qa_system.faiss_index = faiss.read_index(faiss_index_path)
            
            # Load passage IDs
            with open(passage_ids_path, 'r') as f:
                qa_system.indexed_passage_ids = json.load(f)
                
            # Load embeddings if available
            if os.path.exists(embeddings_path):
                qa_system.passage_embeddings = torch.load(embeddings_path, map_location=device, weights_only=True)
            
            # Load retriever model
            qa_system.retriever = SentenceTransformer(TRANSFORMER_NAME)
            qa_system.retriever.to(device)
            
            logger.info(f"FAISS index and related components loaded from {index_dir}")
    
    logger.info("System components loaded successfully")
    return True