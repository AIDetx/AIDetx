import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

OUTPUT = "src/other/alphabets_search"
# create folder if it does not exist
Path(OUTPUT).mkdir(parents=True, exist_ok=True)

DATASETS = ["data1", "data2"]
DATASETS_NAMES = ["AI-human-text", "HC3"]
BEST_K = 8
BEST_ALPHA = 0.5

alphabets = [
    "src/other/alphabets_search/alphabet1.txt",
    "src/other/alphabets_search/alphabet2.txt",
    "src/other/alphabets_search/alphabet3.txt",
    "src/other/alphabets_search/alphabet4.txt",
]

colls = ["dataset", "file", "alphabet", "samples", "hits", "misses"]
df = pd.DataFrame(columns=colls)

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--graphics_only', help='Generate only graphics', action='store_true', default=False)
args = parser.parse_args()

def calculate_f1_score(human_correct, human_incorrect, ai_correct, ai_incorrect):
    # F1 score
    human_precision = human_correct / (human_correct + ai_incorrect)
    human_recall = human_correct / (human_correct + human_incorrect)
    human_f1 = 2 * human_precision * human_recall / (human_precision + human_recall)

    ai_precision = ai_correct / (ai_correct + human_incorrect)
    ai_recall = ai_correct / (ai_correct + ai_incorrect)
    ai_f1 = 2 * ai_precision * ai_recall / (ai_precision + ai_recall)

    f1 = (human_f1 + ai_f1) / 2
    
    return f1

def calculate_accuracy(human_correct, human_incorrect, ai_correct, ai_incorrect):
    # Accuracy
    total_samples = human_correct + human_incorrect + ai_correct + ai_incorrect
    accuracy = (human_correct + ai_correct) / total_samples
    return accuracy

def model_cmd(dataset, k, alphabet, i):
    return f"./bin/train -h data/{dataset}/human_train.txt -g data/{dataset}/ai_train.txt -k {k} -o models/alphabet_search/{dataset}_{i} -a {alphabet}"

def validate_cmd(dataset, j, alpha, type_):
    return f"./bin/was_chatted -m models/alphabet_search/{dataset}_{j} -a {alpha} -d data/{dataset}/{type_}_val.txt"

def plot_alphabet_comparison(alphabets_results, datasets, output):
    plt.figure(figsize=(10, 6))
    plt.title(f"Performance of the model with k={BEST_K} and alpha={BEST_ALPHA} using different alphabets")
    plt.xlabel("Alphabet")
    plt.ylabel("F1 Score")
    
    for i, alphabet in enumerate(alphabets_results):
        plt.bar(np.arange(len(datasets)) + i * 0.15, [alphabets_results[alphabet][dataset] for dataset in datasets[::-1]], width=0.15, label=f"{alphabet}")
    plt.xticks(np.arange(len(datasets)) + (len(alphabets_results) - 1) * 0.15 / 2, DATASETS_NAMES[::-1])
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{output}")

if __name__ == "__main__":

    if not args.graphics_only:
        # create models
        for dataset in DATASETS:
            for i, a in enumerate(alphabets):
                if Path(f"models/alphabet_search/{dataset}_{i}").exists():
                    continue
                cmd = model_cmd(dataset, BEST_K, a, i)
                print(cmd)
                subprocess.run(cmd, shell=True)

        # validate models
        i = 0
        for dataset in DATASETS:
            for j, a in enumerate(alphabets):
                for type_ in ["human", "ai"]:
                    cmd = validate_cmd(dataset, j, BEST_ALPHA, type_)
                    print(cmd)
                    result = subprocess.run(cmd, shell=True, capture_output=True)
                    print(result.stdout.decode("utf-8"))
                    result = result.stdout.decode("utf-8").split("\n")[:-1]
                    print("Result:", result)
                    n_samples = int(result[-3].split(":")[1])
                    human_hits = int(result[-2].split(":")[1].split(",")[0])
                    hits = human_hits if type_ == "human" else n_samples - human_hits
                    misses = n_samples - hits
                    
                    with open(a, "r") as f:
                        alphabet_text = f.read()
                    
                    df.loc[i] = [dataset, type_, alphabet_text, n_samples, hits, misses]
                    i += 1

        df.to_csv(f"{OUTPUT}/alphabet_search.csv", index=False)

    
    # Load the data from the csv
    data = pd.read_csv(f"{OUTPUT}/alphabet_search.csv")
    
    datasets = data['dataset'].unique()
    unique_alphabets = data['alphabet'].unique()
    
    alphabets_results = {alphab: {} for alphab in unique_alphabets }
    for data_index, row in data.iterrows():
        # if row is odd
        if (data_index % 2 == 1):
            continue
        
        # get next row
        next_row = data.iloc[data_index + 1]
        
        # row is human and next row is ai
        f1_score = calculate_f1_score(row['hits'], row['misses'], next_row['hits'], next_row['misses'])
        accuracy = calculate_accuracy(row['hits'], row['misses'], next_row['hits'], next_row['misses'])
        print("dataset:", row["dataset"], "alphabet: ", row['alphabet'], "f1_score: ", f1_score, "accuracy: ", accuracy)
        
        alphabets_results[row['alphabet']][row['dataset']] = f1_score
        
    
    plot_alphabet_comparison(alphabets_results, datasets, f"{OUTPUT}/alphabet_search.png")
    
    


