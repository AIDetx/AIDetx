from datasets import load_dataset
from itertools import chain
import random
import argparse
import re
from pathlib import Path

def save_samples(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(sample + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test", help="Test size", type=float, default=0.1)
parser.add_argument("--human_train", help="Human train data output", type=str, default="data/human_train.txt")
parser.add_argument("--human_test", help="Human test data output", type=str, default="data/human_test.txt")
parser.add_argument("--ai_train", help="AI train data output", type=str, default="data/ai_train.txt")
parser.add_argument("--ai_test", help="AI test data output", type=str, default="data/ai_test.txt")
args = parser.parse_args()


random.seed(args.seed)
dataset = load_dataset("andythetechnerd03/AI-human-text")

human_samples = []
ai_samples = []

for sample in chain(dataset["train"], dataset["test"]):
    text = re.sub(r"[\n\r]", " ", sample["text"].strip())
    if len(text) < 100:
        continue
    if sample["generated"] == 0:
        human_samples.append(text)
    else:
        ai_samples.append(text)

random.shuffle(human_samples)
random.shuffle(ai_samples)

ai_total_chars = sum(len(text) for text in ai_samples)

i = 0
n_chars = 0
for text in human_samples:
    i += 1
    n_chars += len(text)
    if n_chars >= ai_total_chars:
        break
human_samples = human_samples[:i]

print(f"Human samples: {len(human_samples)}, chars:{n_chars}")
print(f"AI samples: {len(ai_samples)}, chars:{ai_total_chars}")

human_train = human_samples[:int(len(human_samples) * (1 - args.test))]
human_test = human_samples[int(len(human_samples) * (1 - args.test)):]
ai_train = ai_samples[:int(len(ai_samples) * (1 - args.test))]
ai_test = ai_samples[int(len(ai_samples) * (1 - args.test)):]

save_samples(human_train, args.human_train)
save_samples(human_test, args.human_test)
save_samples(ai_train, args.ai_train)
save_samples(ai_test, args.ai_test)