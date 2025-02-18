import os
from transformers import BartTokenizer
from datasets import load_from_disk


def tokenize_dataset(dataset_path="data/raw/dataset", model_name="facebook/bart-base",save_path="data/processed/dataset", max_length=512):
    """Tokenizza il dataset (question e answer) e lo salva su disco."""

    # Caricare il dataset
    dataset = load_from_disk(dataset_path)

    # Inizializzare il tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        """Tokenizza batch di question e answer garantendo coerenza nelle lunghezze."""

        # Tokenizzazione delle domande
        tokenized_inputs = tokenizer(
            examples["question"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        # Controllo e tokenizzazione delle risposte
        if "answer" in examples and isinstance(examples["answer"], list):
            answers = [ans if isinstance(ans, str) else "" for ans in examples["answer"]]
        else:
            answers = [""] * len(examples["question"])  # Se non ci sono risposte, riempi con stringhe vuote

        tokenized_answers = tokenizer(
            answers,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        # Aggiunta delle labels con padding coerente
        labels = tokenized_answers["input_ids"]
        labels = [[label if label != tokenizer.pad_token_id else -100 for label in ans] for ans in labels]

        # Aggiungere "labels" ai token
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    # Applicare la tokenizzazione all'intero dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Creare la cartella se non esiste
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Salvare il dataset tokenizzato
    tokenized_dataset.save_to_disk(save_path)
    print(f" Dataset tokenizzato e salvato in {save_path}")

    return tokenized_dataset
