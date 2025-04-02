import os

from config import DEFAULT_MODELS_DIR, logger
from qa_system import BioAsqQASystem
from utils.grid_search import tune_bm25_params

def build_system_from_scratch(qa_system):
    """Build the QA system from scratch"""
    # Load and explore dataset
    qa_system.load_dataset()
    qa_system.explore_dataset()
    
    # Preprocess data
    qa_system.preprocess_data()
    
    # Set up baseline model
    qa_system.setup_baseline()
    
    # Fine-tune if enabled
    if qa_system.use_fine_tuning:
        qa_system.fine_tune(num_train_epochs=3)
    
    # Build and optimize retrieval components if RAG is enabled
    if qa_system.use_rag:
        # Tune BM25 parameters for better retrieval
        best_k1, best_b = qa_system.tune_bm25_params(
            split="test", 
            k=5, 
            sample_size=50, 
            k1_values=[1.2, 1.5, 1.8], 
            b_values=[0.5, 0.7, 0.9]
        )
        logger.info(f"Optimal BM25 parameters: k1={best_k1}, b={best_b}")
    
    # Apply PEFT if enabled
    if qa_system.use_peft:
        qa_system.apply_peft(num_train_epochs=3)

def main():
    """Main function to run the BioAsq Q&A system"""
    # System settings
    models_dir = DEFAULT_MODELS_DIR
    saved_system_path = os.path.join(models_dir, "latest_system")
    force_retrain = False  # Set to True to force retraining
    
    # Initialize the system with desired components
    qa_system = BioAsqQASystem(
        use_fine_tuning=True,
        use_peft=False,  # Set to True to enable PEFT
        use_rag=True,
        models_dir=models_dir
    )
    
    # Check if we can load a saved system
    if os.path.exists(saved_system_path) and not force_retrain:
        logger.info(f"Found saved system at {saved_system_path}, attempting to load...")
        if qa_system.load_all_components(saved_system_path):
            logger.info("Successfully loaded saved system components")
            
            # We still need to load the dataset for evaluation
            qa_system.load_dataset()
        else:
            logger.warning("Failed to load saved system, will build from scratch")
            build_system_from_scratch(qa_system)
    else:
        logger.info("No saved system found or force retrain enabled, building from scratch")
        build_system_from_scratch(qa_system)
    
    # Evaluate the system
    results = qa_system.evaluate("test")
    
    # Evaluate retrieval if RAG is enabled
    if qa_system.use_rag:
        retrieval_metrics = qa_system.evaluate_retrieval("test", k=10, use_hybrid=True)
    
    # Example Q&A
    question = "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
    answer = qa_system.generate_answer(question, use_hybrid=True)
    print(f"\nQuestion: {question}")
    print(f"Generated Answer: {answer}")
    
    # Save the complete system for future use
    qa_system.save_all_components(saved_system_path)
    logger.info(f"System saved to {saved_system_path} for future use")

if __name__ == "__main__":
    main()