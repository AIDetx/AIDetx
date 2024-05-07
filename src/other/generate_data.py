from datasets import load_dataset
from itertools import chain
import random
import argparse
import re
from pathlib import Path

DATASET1 = "andythetechnerd03/AI-human-text"
DATASET2 = "Hello-SimpleAI/HC3"

def save_samples(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(sample + "\n")


def split_samples(samples, name):
    train_slice = [0, int(len(samples) * (1 - args.test - args.val))]
    val_slice = [train_slice[1], int(len(samples) * (1 - args.test))]
    test_slice = [val_slice[1], len(samples)]

    print(f"Train: [{train_slice[0]}:{train_slice[1]}[")
    print(f"Val: [{val_slice[0]}:{val_slice[1]}[")
    print(f"Test: [{test_slice[0]}:{test_slice[1]}[")

    save_samples(samples[train_slice[0]:train_slice[1]], f"data/{name}_train.txt")
    save_samples(samples[val_slice[0]:val_slice[1]], f"data/{name}_val.txt")
    save_samples(samples[test_slice[0]:test_slice[1]], f"data/{name}_test.txt")



def get_samples1():
    dataset = load_dataset(DATASET1)
    diff = set()
    human_samples = []
    ai_samples = []

    for sample in chain(dataset["train"], dataset["test"]):
        text = sample["text"].strip()
        if len(text) < args.min_text_length or text in diff:
            continue
        diff.add(text)
        text = re.sub(r"[\n\r]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        if sample["generated"] == 0:
            human_samples.append(text)
        else:
            ai_samples.append(text)

    random.shuffle(human_samples)
    random.shuffle(ai_samples)
    return human_samples, ai_samples


def get_samples2():
    dataset = load_dataset(DATASET2, name="all")
    diff = set()
    human_samples = []
    ai_samples = []

    for sample in dataset["train"]:
        for texts, lst in [(sample["human_answers"], human_samples), (sample["chatgpt_answers"], ai_samples)]:
            for text in texts:
                text = text.strip()
                if len(text) < args.min_text_length or text in diff:
                    continue
                diff.add(text)
                text = re.sub(r"[\n\r]+", " ", text)
                text = re.sub(r"\s+", " ", text)
                lst.append(text)

    random.shuffle(human_samples)
    random.shuffle(ai_samples)
    return human_samples, ai_samples


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=42)
parser.add_argument("-t", "--test", help="Test size", type=float, default=0.1)
parser.add_argument("-v", "--val", help="Validation size", type=float, default=0.1)
parser.add_argument("-m", "--min_text_length", help="Minimum text length", type=int, default=100)
args = parser.parse_args()

random.seed(args.seed)

for fn, dataset in [(get_samples1, "data1"), (get_samples2, "data2")]:
    if Path(f"data/{dataset}").exists():
        continue
    human_samples, ai_samples = fn()
    ai_total_chars = sum(len(text) for text in ai_samples)
    human_total_chars = sum(len(text) for text in human_samples)

    chars = min(ai_total_chars, human_total_chars)
    if ai_total_chars < human_total_chars:
        human_samples = human_samples[:int(chars / human_total_chars * len(human_samples))]
    else:
        ai_samples = ai_samples[:int(chars / ai_total_chars * len(ai_samples))]

    print(f"\nDataset: {dataset}")
    print(f"Human samples: {len(human_samples)}, chars:{sum(len(text) for text in human_samples)}")
    print(f"AI samples: {len(ai_samples)}, chars:{sum(len(text) for text in ai_samples)}")

    split_samples(human_samples, f"{dataset}/human")
    split_samples(ai_samples, f"{dataset}/ai")