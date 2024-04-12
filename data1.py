from datasets import load_dataset

from unidecode import unidecode

# print(unidecode("a😆a"))
# exit()

# print(ord("😆"))
# print(chr(128518))
# exit()

dataset = load_dataset("andythetechnerd03/AI-human-text")

print(dataset)

# letters = set()

# for sample in dataset["train"]:
#     for letter in sample["text"]:
#         letters.add(letter)
# for sample in dataset["test"]:
#     for letter in sample["text"]:
#         letters.add(letter)
# print(letters)

# alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# for letter in alphanumeric:
#     if letter not in letters:
#         print(letter)

# human_samples = sum(s["generated"] == 0 for s in dataset["train"])
# ai_samples = sum(s["generated"] == 1 for s in dataset["train"])
# total = human_samples + ai_samples
# print(f"Dataset size: {total}")
# print(f"Human samples: {human_samples}, {human_samples/total * 100:.2f}%")
# print(f"AI samples: {ai_samples}, {ai_samples/total * 100:.2f}%")

# human_samples = sum(s["generated"] == 0 and all(ord(c) < 128 for c in s["text"]) for s in dataset["train"])
# ai_samples = sum(s["generated"] == 1 and all(ord(c) < 128 for c in s["text"]) for s in dataset["train"])
# total = human_samples + ai_samples
# print(f"Dataset size: {total}")
# print(f"Human samples: {human_samples}, {human_samples/total * 100:.2f}%")
# print(f"AI samples: {ai_samples}, {ai_samples/total * 100:.2f}%")

human_text = sum(len(s["text"].strip()) for s in dataset["train"] if s["generated"] == 0)
ai_text = sum(len(s["text"].strip()) for s in dataset["train"] if s["generated"] == 1)

total_text = human_text + ai_text
print(f"Human text length: {human_text}, {human_text/total_text * 100:.2f}%")
print(f"AI text length: {ai_text}, {ai_text/total_text * 100:.2f}%")