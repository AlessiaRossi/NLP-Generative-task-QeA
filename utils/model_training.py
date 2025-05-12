import os
import json
from datetime import datetime
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from config import logger, device, MODEL_NAME
from utils.monitoring import plot_training_history, analyze_learning_curve

def fine_tune_model(qa_system, output_dir="./fine_tuned_model", num_train_epochs=15, force_train=False):
    """Fine-tune the model on the BioAsq dataset

    Args:
        qa_system: The QA system instance
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs (default increased to 15 for proper early stopping)
        force_train: If True, force re-training even if a model already exists

    Returns:
        None
    """
    if not qa_system.use_fine_tuning:
        logger.info("Fine-tuning is disabled. Skipping.")
        return
    
    # Create full output path
    full_output_dir = os.path.join(qa_system.models_dir, output_dir)
    
    # Check if fine-tuned model already exists
    if os.path.exists(full_output_dir) and os.path.isdir(full_output_dir) and not force_train:
        logger.info(f"Loading existing fine-tuned model from {full_output_dir}")
        qa_system.tokenizer = AutoTokenizer.from_pretrained(full_output_dir)
        if qa_system.use_peft:
            # For PEFT models, load differently
            qa_system.model = PeftModel.from_pretrained(
                qa_system.model,  # Base model
                full_output_dir,  # PEFT adapter weights
            )
        else:
            qa_system.model = AutoModelForSeq2SeqLM.from_pretrained(full_output_dir)
        
        qa_system.model.to(device)
        logger.info("Fine-tuned model loaded successfully")
        return
        
    logger.info("Preparing data for fine-tuning...")
    
    # Look for a training split - try common names or use the first available split
    train_split_names = ["train", "training", "finetune"]
    train_split = None
    
    # First try common training split names
    for split_name in train_split_names:
        if split_name in qa_system.dataset:
            train_split = split_name
            break
    
    # If no train split found, use the first available split that's not "corpus"
    if train_split is None:
        available_splits = [split for split in qa_system.dataset.keys() if split != "corpus"]
        if available_splits:
            train_split = available_splits[0]
            logger.warning(f"No standard training split found. Using '{train_split}' split for training.")
        else:
            logger.error("No suitable data split found for fine-tuning. Aborting.")
            return
    
    # Get training data
    train_data = qa_system.dataset[train_split]
    logger.info(f"Using '{train_split}' split with {len(train_data)} examples for fine-tuning")
    
    # Create a validation split from the training data
    train_val_dict = train_data.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_dict["train"]
    eval_dataset = train_val_dict["test"]
    logger.info(f"Split into {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
    
    # Preprocess function to prepare inputs and labels
    def preprocess_function(examples):
        """Preprocess the examples for training"""
        inputs = ["question: " + q for q in examples["question"]]
        targets = examples["answer"]
        
        # Add context for RAG if available
        if qa_system.use_rag and "relevant_passage_ids" in examples:
            for i, (question, passage_ids_str) in enumerate(zip(examples["question"], examples["relevant_passage_ids"])):
                context = []
                # Parse the string representation of the list into an actual list
                try:
                    passage_ids = json.loads(passage_ids_str.replace("'", '"'))
                    for pid in passage_ids:
                        pid_normalized = qa_system._normalize_id(pid)
                        if pid_normalized in qa_system.passage_dict:
                            context.append(qa_system.passage_dict[pid_normalized])
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(f"Could not parse passage IDs: {passage_ids_str}")
                
                if context:
                    inputs[i] = f"question: {question} context: {' '.join(context)}"

        # Tokenize inputs
        model_inputs = qa_system.tokenizer(
            inputs, max_length=512, padding="max_length", truncation=True
        )
        
        # Tokenize targets
        with qa_system.tokenizer.as_target_tokenizer():
            labels = qa_system.tokenizer(
                targets, max_length=128, padding="max_length", truncation=True
            ).input_ids
            
        model_inputs["labels"] = labels
        return model_inputs
    
    tokenized_train_data = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_data = eval_dataset.map(preprocess_function, batched=True)
    
    # Configure training arguments with early stopping
    training_args = TrainingArguments(
        output_dir=full_output_dir,
        seed=42,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        logging_strategy="epoch",
        report_to=["tensorboard"],
    )

    # Add early stopping with increased patience
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)
    
    # Initialize trainer with both train and eval datasets
    trainer = Trainer(
        model=qa_system.model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        tokenizer=qa_system.tokenizer,
        callbacks=[early_stopping_callback],
    )
    
    # Start training
    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning completed.")
    
    # Save the model
    logger.info(f"Saving fine-tuned model to {full_output_dir}")
    trainer.save_model(full_output_dir)
    qa_system.tokenizer.save_pretrained(full_output_dir)
    
    # Generate training visualizations and analysis
    plot_path = plot_training_history(trainer, full_output_dir)
    if plot_path:
        logger.info(f"Training history plot saved to {plot_path}")
    
    analysis = analyze_learning_curve(trainer, full_output_dir)
    if analysis and 'error' not in analysis:
        logger.info(f"Learning curve analysis: {analysis['recommendation']}")
    
    # Also save training metadata with analysis results
    with open(os.path.join(full_output_dir, "training_metadata.json"), 'w') as f:
        metadata = {
            "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": num_train_epochs,
            "use_peft": qa_system.use_peft,
            "use_rag": qa_system.use_rag,
            "base_model": MODEL_NAME,
            "actual_epochs_trained": len(trainer.state.log_history),
            "best_epoch": analysis.get("min_loss_epoch") if analysis and 'error' not in analysis else None,
            "early_stopping": {
                "patience": 5,
                "recommendation": analysis.get("recommendation") if analysis and 'error' not in analysis else None
            }
        }
        json.dump(metadata, f, indent=2)

def apply_peft(qa_system, output_dir="./peft_model", num_train_epochs=15, force_train=False):
    """Apply Parameter-Efficient Fine-Tuning using LoRA with increased epochs for proper early stopping"""
    if not qa_system.use_peft:
        logger.info("PEFT is disabled. Skipping.")
        return
    
    # Create full output path
    full_output_dir = os.path.join(qa_system.models_dir, output_dir)
    
    # Check if PEFT model already exists
    if os.path.exists(full_output_dir) and os.path.isdir(full_output_dir) and not force_train:
        logger.info(f"Loading existing PEFT model from {full_output_dir}")
        qa_system.model = PeftModel.from_pretrained(
            qa_system.model,  # The base model
            full_output_dir,  # PEFT adapter weights
        )
        qa_system.model.to(device)
        logger.info("PEFT model loaded successfully")
        return
        
    logger.info("Applying LoRA for Parameter-Efficient Fine-Tuning...")
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Convert model to PEFT model
    qa_system.model = get_peft_model(qa_system.model, peft_config)
    
    # Display trainable parameters info
    trainable_params = sum(p.numel() for p in qa_system.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in qa_system.model.parameters())
    logger.info(f"Trainable params: {trainable_params} ({100 * trainable_params / total_params:.2f}% of total)")
    
    # Now fine-tune with the PEFT model (using the same process as fine_tune)
    fine_tune_model(qa_system, output_dir=output_dir, num_train_epochs=num_train_epochs, force_train=force_train)