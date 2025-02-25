import os
from transformers import BartTokenizer
from datasets import load_dataset, load_from_disk
from transformers import (T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback)

def tokenize_dataset(dataset_path=RAW_DATASET_PATH, model_name="google-t5/t5-small", save_path=PROCESSED_DATASET_PATH, max_length=512):
    """Tokenizza il dataset (question e answer) e lo salva su disco."""
    dataset = load_from_disk(dataset_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    def tokenize_function(examples):
        inputs = [f"question: {q} context: {' '.join(p)}" for q, p in zip(examples["question"], examples["relevant_passage_ids"])]
        tokenized_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        answers = [ans if isinstance(ans, str) else "" for ans in examples.get("answer", [""] * len(examples["question"]))]
        tokenized_answers = tokenizer(
            answers,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        labels = tokenized_answers["input_ids"]
        labels = [[label if label != tokenizer.pad_token_id else -100 for label in ans] for ans in labels]

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    first_split = list(dataset.keys())[0]
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset[first_split].column_names
    )
    print(f"Split disponibili nel dataset caricato: {dataset.keys()}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenized_dataset.save_to_disk(save_path)
    print(f"Dataset tokenizzato e salvato in {save_path}")
    return tokenized_dataset
