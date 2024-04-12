from datasets import load_dataset
from itertools import chain
import random
import time
from unidecode import unidecode

# seed
random.seed(42)

dataset = load_dataset("andythetechnerd03/AI-human-text")

print(dataset)

# string array

human_samples = []
ai_samples = []

start = time.time()
for sample in chain(dataset["train"], dataset["test"]):
    text = unidecode(sample["text"].strip()).upper()
    # text = sample["text"].strip()
    # if not all(ord(c) <= 122 and ord(c) >= 32 for c in text):
    #     continue
    # text = text.upper()
    if sample["generated"] == 0:
        human_samples.append(text)
    else:
        ai_samples.append(text)
end = time.time()
print(f"Time taken: {end - start:.2f}s")


# shuffle
start = time.time()
random.shuffle(human_samples)
random.shuffle(ai_samples)
end = time.time()
print(f"Time taken: {end - start:.2f}s")

print(f"Human samples: {len(human_samples)}")
print(f"AI samples: {len(ai_samples)}")

chars = set()
for text in chain(human_samples, ai_samples):
    for char in text:
        chars.add(char)

ascii_chars = sorted(ord(c) for c in chars)
print(ascii_chars)


