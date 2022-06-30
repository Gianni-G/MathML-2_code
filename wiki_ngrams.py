import datasets
from pathlib import Path
from collections import Counter
from tqdm import tqdm, trange
import util

datasets.config.HF_DATASETS_CACHE = Path("/cluster/scratch/gjuan/.cache/huggingface/datasets")

wiki = datasets.load_dataset("wikipedia", "20220301.en")

wiki_train = wiki["train"]


thres = 10
ng_n = 4

ngrams = Counter()
for example in tqdm(wiki_train):
    for ng in zip(*[example["text"][i:] for i in range(ng_n)]):
        ngrams[ng] += 1

ngrams = {"".join(tup) : freq for tup, freq in ngrams.most_common() if freq>=thres}

util.save_file(ngrams, f"ngrams/{ng_n}_{thres}.json")