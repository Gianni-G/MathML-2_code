import datasets
from pathlib import Path
from collections import Counter
from tqdm import tqdm, trange
import semiolog as slg

datasets.config.HF_DATASETS_CACHE = Path("/cluster/scratch/gjuan/.cache/huggingface/datasets")

wiki = datasets.load_dataset("wikipedia", "20220301.en")

wiki_train = wiki["train"]


def normalize(text):
    text = text.lower()
    norm_text = ""
    for c in text:
        if c not in punctuation:
            norm_text = norm_text + c
    return norm_text

text = wiki_train[1]["text"]
normalize(text)