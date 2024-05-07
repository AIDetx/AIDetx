import os
import re
import subprocess
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
plt.rcParams.update({'font.size': 13})

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='file with hyperparameter results', default='src/other/grid_search.csv')
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


if __name__ == "__main__":
    
    # Load the data from the csv
    data = pd.read_csv(args.file)
    
    ks = data['k'].unique()
    ks_index = {k: i for i, k in enumerate(ks)}
    alphas = data['alpha'].unique()
    
    print(data)
    print(ks)
    print(alphas)
    
    # iterate over data
    accs = [[] for _ in range(len(ks))]
    evaluation_times = [[] for _ in range(len(ks))]
    for data_index, row in data.iterrows():
        # if row is odd
        if data_index % 2 == 1:
            continue
        
        # get next row
        next_row = data.iloc[data_index + 1]
        
        # row is human and next row is ai
        f1_score = calculate_f1_score(row['hits'], row['misses'], next_row['hits'], next_row['misses'])
        
        accs[ks_index[row['k']]].append(f1_score)
        evaluation_times[ks_index[row['k']]].append(next_row['time'] + row['time'])
        
    
    # plot the results using x as k, y as F1 Score and different colors for each alpha
    plt.figure(figsize=(10, 6))
    for i, alpha in enumerate(alphas):
        alphas_acc = [accs[j][i] for j in range(len(ks))]
        plt.plot(ks, alphas_acc, label=f"alpha={alpha}")
    plt.xlabel("k")
    plt.ylabel("f1 Score")
    plt.title("Hyperparameter Tunning")
    plt.legend()
    plt.savefig("src/other/hyperparameter.png")
    
    plt.clf()
    
    # plot the results using x as k, y as evaluation time and different colors for each alpha
    plt.figure(figsize=(10, 6))
    for i, alpha in enumerate(alphas):
        alphas_time = [evaluation_times[j][i] for j in range(len(ks))]
        plt.plot(ks, alphas_time, label=f"alpha={alpha}")
    plt.xlabel("k")
    plt.ylabel("Evaluation Time (s)")
    plt.title("Evaluation Time in function of the hyperparameters")
    plt.legend()
    plt.savefig("src/other/hyperparameter_evaluation_time.png")
    
    
    