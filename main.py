import os
from datasets import load_dataset
from tokeniza import tokenize_dataset
from train import train_model

def download_dataset(dataset_name="rag-datasets/rag-mini-bioasq", config_name="question-answer-passages", save_path="data/raw/dataset"):
    os.makedirs(save_path, exist_ok=True)
    dataset = load_dataset(dataset_name, config_name)

    # Stampiamo le chiavi disponibili nel dataset originale
    print(f"Splits disponibili nel dataset originale: {dataset.keys()}")

    dataset.save_to_disk(save_path)
    print(f" Dataset originale scaricato e salvato in {save_path}")


if __name__ == "__main__":
    download_dataset()
    print(" **Avvio della tokenizzazione..**")
    raw_dataset_path = "data/raw/dataset"
    processed_dataset_path = "data/processed/dataset"
    tokenize_dataset(dataset_path=raw_dataset_path, save_path=processed_dataset_path)
    print("**Tokenizzazione completata!**")
    print(" **Avvio del fine-tuning...**")
    train_model(dataset_path=processed_dataset_path)
    print(" **Fine-tuning completato!**")
