import os
from transformers import BartTokenizer
from datasets import load_from_disk


def tokenize_dataset(dataset_path="data/raw/dataset", model_name="facebook/bart-base",
                     save_path="data/processed/dataset"):
    """Carica, tokenizza e salva il dataset locale."""

    # Caricare il dataset salvato da main.py
    dataset = load_from_disk(dataset_path)

    # Inizializzare il tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        """Tokenizza domande e risposte."""
        tokenized_inputs = tokenizer(
            examples["question"],  # Tokenizza la domanda
            padding="max_length",
            truncation=True
        )
        tokenized_answers = tokenizer(
            examples["answer"],  # Tokenizza la risposta
            padding="max_length",
            truncation=True
        )
        tokenized_inputs["labels"] = tokenized_answers["input_ids"]
        return tokenized_inputs

    # Applicare la tokenizzazione a tutto il dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Creare la cartella se non esiste
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Salvare il dataset tokenizzato
    tokenized_dataset.save_to_disk(save_path)
    print(f"\n Dataset tokenizzato e salvato in {save_path}")

    # Mostrare alcuni esempi dal dataset tokenizzato
    print("\n Esempi di dati tokenizzati:")

    # Controllare se il dataset ha "test" o "train"
    split_name = "test" if "test" in tokenized_dataset else "train"

    for i in range(min(3, len(tokenized_dataset[split_name]))):  # Mostra fino a 3 esempi
        print(f"\n  Esempio {i + 1}:")
        print(f" input_ids: {tokenized_dataset[split_name][i]['input_ids'][:20]}...")  # Primi 20 token
        print(f" labels: {tokenized_dataset[split_name][i]['labels'][:20]}...")

    return tokenized_dataset



