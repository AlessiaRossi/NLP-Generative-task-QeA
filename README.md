# Biomedical Question Answering with Retrieval-Augmented Generation (RAG)
##  Index

- [Project Description](#project-description)
- [Environment Setup](#environment-setup)
- [Data Source](#data-source)
- [Pipeline Overview](#pipeline-overview)
- [Evaluation](#Evaluation)

---
## Project Description


This project presents a modular Biomedical Question Answering (BioQA) system built on the **Retrieval-Augmented Generation (RAG)** paradigm. The system answers factoid-style biomedical questions by retrieving relevant context passages and generating answers using fine-tuned or external models.

It is structured to support various research and evaluation scenarios, including fine-tuning, parameter-efficient training, LLM prompting, and retriever optimization.

---

##  Techniques Used

The following techniques are implemented and tested in this project:

-  **Fine-tuning** of small to mid-sized encoder-decoder models such as **BioBART** and **T5-base**, including **early stopping**.
-  **Prompting** a very large language model (**GPT-4**) with retrieved passages as context, without any additional training.
-  **PEFT (Parameter-Efficient Fine-Tuning)** via **LoRA**, applied to attention projection layers for low-resource adaptation.
-  **RAG (Retrieval-Augmented Generation)** using both:
  - **Semantic retrieval** with SentenceTransformers + FAISS
  - **Lexical retrieval** using BM25
  - **Hybrid merging** with reranking via CrossEncoder
  - **BM25 grid search** to tune `k1` and `b` for optimal retrieval

---

## Environment Setup

1. **Clone the repository**
```bash
git clone https://github.com/AlessiaRossi/NLP-Generative-task-QeA
cd NLP-Generative-task-QeA
```

2**Install dependencies**
```bash
pip install -r requirements.txt
```
3**Set OpenAI API key**
```bash
export OPENAI_API_KEY="your-key-here"

```
---
##  Project Structure
 The project structure refers to the organization of files and folders within the application. It plays a crucial role in maintaining the application, adding new features, and facilitating collaboration. A well-defined project structure enhances code readability and navigation.
```plaintext
├── main.py                  # Main pipeline entry point
├── qa_system.py             # Core Q&A class (BioAsqQASystem)
├── config.py                # Default config and model names
├── requirements.txt         # Python dependencies
├── utils/
│   ├── dataset_manager.py   # Dataset loading and preprocessing
│   ├── evaluation.py        # BLEU, ROUGE, Recall/Precision calculations
│   ├── retrieval.py         # Query expansion and ID handling
│   ├── grid_search.py       # BM25 grid search implementation
│   └── model_training.py    # Fine-tuning and PEFT logic
```
---
## Data Source

The dataset used is the **BioASQ mini dataset**, loaded via [HuggingFace Datasets](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq):

- **Language**: English  
- **Domain**: Biomedical and clinical research  
- **Subsets**:
  - `text-corpus`: Biomedical passages  
  - `question-answer-passages`: Factoid Q&A pairs

The system supports full preprocessing, ID normalization, and indexing of the passage corpus for downstream retrieval tasks.

---

## Pipeline Overview


### 1. Fine-Tuning a “Small” Model 

We fine-tuned models like **BioBART** (`GanjinZero/biobart-v2-base`) on the BioASQ factoid dataset. Key implementation details include:

- Use of **retrieved context passages** when Retrieval-Augmented Generation (RAG) is active
- Training with **HuggingFace Trainer**, enabling:
  - **Early stopping** via `EarlyStoppingCallback`
  - Epoch-wise evaluation and checkpointing
- Fine-tuning aligns the model with biomedical terminology and QA structure

This approach allows the models to adapt more closely to biomedical terminology and structure.

### 2. Prompting a Very Large Language Model (LLM)
The system supports **prompt-based inference** using external LLMs like **GPT-4**, accessed through the OpenAI API.
### 3. Parameter-Efficient Fine-Tuning (PEFT with LoRA)

To reduce training time and memory usage, we implemented **Low-Rank Adaptation (LoRA)** using the HuggingFace `peft` library.

- LoRA adapters added to attention projections (`q_proj`, `v_proj`)
- Only adapters are trained, base model is frozen
- Compatible with the existing fine-tuning loop
- Benefits:
- Lower memory footprint
- Faster convergence
- Lightweight deployment

### 4. Retrieval-Augmented Generation (RAG)

We implemented a RAG-style pipeline that retrieves relevant context passages before answer generation.

- **Semantic Retrieval**: SentenceTransformers (`all-MiniLM-L6-v2`) + **FAISS**
- **Lexical Retrieval**: **BM25** via rank-bm25
- **Hybrid Strategy**:
- Combine semantic and lexical results
- **CrossEncoder** reranking (`ms-marco-MiniLM-L-6-v2`)
- **Query Expansion**: Biomedical synonyms to improve recall
- **Grid Search over BM25 parameters (k1, b)**:
- Best config (k1=1.6, b=0.75) selected based on Recall@k on validation

---

##  Evaluation

### Answer Generation Evaluation

We used **automatic metrics** to evaluate the fluency and relevance of generated answers:

- **BLEU**: Evaluates n-gram precision
- **ROUGE-1**, **ROUGE-2**, **ROUGE-L**: Measures recall overlap with reference answers

Results (BioBART + all-MiniLM-L6-v2):

| Metric   | Score   |
|----------|---------|
| BLEU     | 0.0803  |
| ROUGE-1  | 0.2956  |
| ROUGE-2  | 0.1337  |
| ROUGE-L  | 0.2378  |

---

### Retrieval Evaluation

Passage retrieval quality was assessed using:

- **Recall@10**: 0.4713
- **Precision@10**: 0.4410

Performance varied across embedding models, with **MiniLM** outperforming larger biomedical-specific encoders (e.g., BiomedBERT, BioLinkBERT).

| Embedding Model               | Recall@10 | Precision@10 |
|------------------------------|-----------|---------------|
| all-MiniLM-L6-v2             | 0.4713    | 0.4410        |
| BiomedBERT                   | 0.1087    | 0.0650        |
| BioLinkBERT                  | 0.0375    | 0.0360        |

---
