# config.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MISTRAL_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"
T5_MODEL = "google/flan-t5-base"

INDEX_PATH_256 = "data/indexes/index_main_256"
INDEX_PATH_512 = "data/indexes/index_main_512"

DEFAULT_INDEX_DIR = INDEX_PATH_256
SIMILARITY_TOP_K = 3