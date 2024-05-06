import os
import re
import subprocess
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt

import numpy as np
plt.rcParams.update({'font.size': 13})

TRAIN = "./bin/train"
WAS_CHATTED = "./bin/was_chatted"
ALPHA = 0.5

STEP = 50
LOWEST_SAMPLE_SIZE = 2364
MAX_NUMBER_OF_SAMPLES = 5000

GRAPHICS_ONLY = True

parser = argparse.ArgumentParser()
parser.add_argument('-ng', '--not_gpt', help='file with human text samples', default='data/human_test.txt')
parser.add_argument('-g', '--gpt', help='file with gpt text samples', default='data/ai_test.txt')
parser.add_argument('-a', '--alphabet', help='file with the alphabet', default='data/alphabet.txt')
parser.add_argument('-k', help='context length', default=5)
parser.add_argument('-n', help='number of sample points for analysis', default=20, type=int)
parser.add_argument('-m', help='max number of samples for analysis', default=5000, type=int)
parser.add_argument('-gg', '--only_graphics', help='only generate graphics', action='store_true', default=False)
parser.add_argument('-hf', '--hyperparameter_file', help='file with hyperparameter results', default='src/other/hyperparameter.csv')
args = parser.parse_args()


def run_train(human_file, ai_file, output, alphabet_file, k):
    command = f"{TRAIN} -h {human_file} -g {ai_file} -a {alphabet_file} -k {k} -o {output}"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)
        result = result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        print("Error running command:", e)


def run_was_chatted(model, data, alpha, name):
    command = f"{WAS_CHATTED} -m {model} -d {data} -a {alpha}"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)
        result = result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        print("Error running command:", e)
    
    result = result.split("\n")
    samples = 0
    correct = 0
    for line in result:
        if line.startswith("Total samples: "):
            samples = line.split(": ")[1]
        elif line.startswith("Samples classified as human: ") and name == "human":
            correct = line.split(": ")[1].split(",")[0]
        elif line.startswith("Samples classified as AI: ") and name == "ai":
            correct = line.split(": ")[1].split(",")[0]

    return int(samples), int(correct)


if __name__ == "__main__":
    
    if not args.only_graphics:
    
        # Get the alphabet
        with open(args.alphabet, "r") as f:
            alphabet = f.read().strip()
        print("alphabet:", alphabet)
        
        # Create the folder that will contain the temporary datasets
        Path("models/temp/hyperparameter").mkdir(parents=True, exist_ok=True)
        
        # Hyperparameter tunning
        # values to test
        ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        alphas = [0.1, 0.2, 0.4, 0.7, 1]
        
        accs = []   # list of list of performance of alpha for each k
        evaluation_times = []   # list of list of evaluation times for each alpha for each k
        model_training_times = []   # list of model training times for each k
        for i, k in enumerate(ks):
            print(f"\nModel {i+1} --> k={k}")
            accs.append([])
            evaluation_times.append([])
            
            # Train the model
            print("Training models...")
            start_model = time.time()
            run_train("data/human_train.txt", "data/ai_train.txt", f"models/temp/hyperparameter/k_{k}", args.alphabet, k)
            end_model = time.time()
            
            model_training_times.append(end_model - start_model)
            print(f"Model training time: {model_training_times[-1]}")
            
            for j, alpha in enumerate(alphas):
                print(f"\n\talpha={alpha}")
            
                print("\tEvaluating model with human samples")
                start_evaluation = time.time()
                samples_human, correct_human = run_was_chatted(f"models/temp/hyperparameter/k_{k}", "data/human_test.txt", alpha, "human")

                print("\tEvaluating model with AI samples")
                samples_ai, correct_ai = run_was_chatted(f"models/temp/hyperparameter/k_{k}", "data/ai_test.txt", alpha, "ai")
                end_evaluation = time.time()

                acc = (correct_human + correct_ai) / (samples_human + samples_ai)
                
                accs[i].append(acc)
                evaluation_times[i].append(end_evaluation - start_evaluation)
                
                print(f"\tAccuracy: {acc}")
        
        
        os.system("rm -r models/temp")
    
        # Store the data on csv
        with open("src/other/hyperparameter.csv", "w") as f:
            f.write("ks=[" + ",".join(map(str, ks)) + "]\n")
            f.write("alphas=[" + ",".join(map(str, alphas)) + "]\n\n")
            
            f.write("model_training_times=" + str(model_training_times) + "\n\n")
            
            f.write("k,alpha,accuracy,evaluation_time\n")
            for i, k in enumerate(ks):
                for j, alpha in enumerate(alphas):
                    f.write(f"{k},{alpha},{accs[i][j]},{evaluation_times[i][j]}\n")
                
    
    # Load the data from the csv
    ks = []
    alphas = []
    with open(args.hyperparameter_file, "r") as f:
        ks = eval(f.readline().strip().split("=")[1])
        alphas = eval(f.readline().strip().split("=")[1])
        
        f.readline()    # skip the empty line
        
        model_training_times = eval(f.readline().strip().split("=")[1])
        
        f.readline()    # skip the empty line
        f.readline()    # skip the header
        
        accs = [[] for _ in range(len(ks))]
        evaluation_times = [[] for _ in range(len(ks))]
        
        # get the index for each value of k
        ks_index = {k: i for i, k in enumerate(ks)}
        
        for line in f:
            k, alpha, acc, evaluation_time = line.strip().split(",")
            k = int(k)
            alpha = float(alpha)
            acc = float(acc)
            evaluation_time = float(evaluation_time)
            
            accs[ks_index[k]].append(acc)
            evaluation_times[ks_index[k]].append(evaluation_time)
            
        
    
    # plot the results using x as k, y as accuracy and different colors for each alpha
    plt.figure(figsize=(10, 6))
    for i, alpha in enumerate(alphas):
        alphas_acc = [accs[j][i] for j in range(len(ks))]
        plt.plot(ks, alphas_acc, label=f"alpha={alpha}")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
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
    plt.ylabel("Evaluation Time")
    plt.title("Evaluation Time in function of the hyperparameters")
    plt.legend()
    plt.savefig("src/other/hyperparameter_evaluation_time.png")
    
    plt.clf()
    
    # plot the results using x as k, y as model training time
    plt.figure(figsize=(10, 6))
    plt.plot(ks, model_training_times)
    plt.xlabel("k")
    plt.ylabel("Model Training Time")
    plt.title("Model Training Time in function of k")
    plt.savefig("src/other/hyperparameter_model_training_time.png")
    
    
    