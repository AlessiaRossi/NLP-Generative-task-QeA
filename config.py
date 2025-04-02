import os
import logging
import torch
import warnings

# Silence messages from absl and tensorflow
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.root.removeHandler(absl.logging._absl_handler)

# Additional logging configuration to silence transformers messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)

# Filter specific warnings
warnings.filterwarnings('ignore', message='.*default tokenizer.*')
warnings.filterwarnings('ignore', message='.*expandable_segments not supported.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.module')

# Setup logging with more detailed format
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "5cd54477ffe79d9ca83fd47ba87a627443c5363a")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Dataset and model names
DATASET_NAME = "rag-datasets/rag-mini-bioasq"
DATASET_SUBSET = ["question-answer-passages", "text-corpus"]
MODEL_NAME = "GanjinZero/biobart-v2-base"
TRANSFORMER_NAME = "all-MiniLM-L6-v2"
ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Default directory for models
DEFAULT_MODELS_DIR = "./models1"