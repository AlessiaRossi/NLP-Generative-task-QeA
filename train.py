import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from evaluate import load as load_metric


def train_model(dataset_path="data/processed/dataset", model_name="facebook/bart-base", output_dir="models/bart-finetuned"):
    """Carica il dataset tokenizzato e avvia il fine-tuning del modello BART."""

    # Verifica se è disponibile una GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricare il dataset tokenizzato
    dataset = load_from_disk(dataset_path)

    # Controllare quali split sono presenti
    available_splits = list(dataset.keys())
    print(f" Available dataset splits: {available_splits}")

    # Se abbiamo solo "test", lo usiamo per creare "train" e "test"
    if "train" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"] if "test" in dataset else None
    elif "test" in dataset:
        print("Il dataset non ha 'train'. Suddivisione automatica dei dati...")

        dataset = dataset["test"].train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        print(f"Nuova suddivisione - Train: {len(train_dataset)}, Test: {len(eval_dataset)}")
    else:
        raise ValueError("Nessun dataset valido trovato. Controlla il processo di tokenizzazione.")

    # Inizializzare il tokenizer e il modello BART
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Definire la funzione di valutazione con ROUGE
    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(np.argmax(logits, axis=-1), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Filtro per evitare token speciali nel calcolo della metrica
        decoded_labels = [[token for token in label if token != -100] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Definire i parametri di training
    training_args = TrainingArguments(
        output_dir="results",
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch" if eval_dataset else "no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="rougeL",
        greater_is_better=True
    )

    # Inizializzare Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Può essere None se il dataset non ha uno split
        compute_metrics=compute_metrics if eval_dataset else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset else []
    )

    # Avviare il training
    trainer.train()

    # Salvare il modello fine-tunato
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nFine-tuning completato! Modello salvato in {output_dir}")
