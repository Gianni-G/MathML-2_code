import datasets
from pathlib import Path
from collections import Counter
from tqdm import tqdm, trange
import util

datasets.config.HF_DATASETS_CACHE = Path("/cluster/scratch/gjuan/.cache/huggingface/datasets")

wiki = datasets.load_dataset("wikipedia", "20220301.en")

wiki_train = wiki["train"]

text = wiki_train[2]["text"]

digits = {str(i) for i in range(10)}
digits_c = digits.union(set([",","."]))

numerals = Counter()
num = ""

for example in tqdm(wiki_train):
    for c in example["text"]:
        if c not in digits_c:
            if num != "":
                numerals[num] += 1
                num = ""
        elif c in digits:
            num = num + c

    if num != "":
        numerals[num] += 1

numerals = {num : freq for num, freq in numerals.most_common()}

util.save_file(numerals, f"ngrams/numerals.json")