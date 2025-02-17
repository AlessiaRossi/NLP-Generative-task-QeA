import os
from datasets import load_dataset
from tokeniza import tokenize_dataset

# File 1: Scaricare il dataset originale e salvarlo
def download_dataset(dataset_name="rag-datasets/rag-mini-bioasq", config_name="question-answer-passages", save_path="data/raw/dataset"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Creazione cartelle necessarie
    dataset = load_dataset(dataset_name, config_name)
    dataset.save_to_disk(save_path)
    print(f"Dataset originale scaricato e salvato in {save_path}")
    print (dataset)

if __name__ == "__main__":
    download_dataset()

# Tokenizzare il dataset direttamente dal main
    print(" **Avvio della tokenizzazione..**")
    raw_dataset_path = "data/raw/dataset"
    processed_dataset_path = "data/processed/dataset"
    tokenize_dataset(dataset_path=raw_dataset_path, save_path=processed_dataset_path)
    print("**Tokenizzazione completata!**")