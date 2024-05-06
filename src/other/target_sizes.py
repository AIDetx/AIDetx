import os
import re
import subprocess
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
plt.rcParams.update({'font.size': 13})

TRAIN = "./bin/train"
WAS_CHATTED = "./bin/was_chatted"
ALPHA = 0.5
K = 5

STEP = 50
LOWEST_SAMPLE_SIZE = 2364
MAX_NUMBER_OF_SAMPLES = 5000

parser = argparse.ArgumentParser()
parser.add_argument('-ng', '--not_gpt', help='file with human text samples', default='data/human_test.txt')
parser.add_argument('-g', '--gpt', help='file with gpt text samples', default='data/ai_test.txt')
parser.add_argument('-a', '--alphabet', help='file with the alphabet', default='data/alphabet.txt')
parser.add_argument('-k', help='context length', default=5)
parser.add_argument('-n', help='number of sample points for analysis', default=20, type=int)
parser.add_argument('-m', help='max number of samples for analysis', default=5000, type=int)
args = parser.parse_args()


# Used to test which values of the sample will be used for the analysis (we want big samples and a lot of them, we got around 5010 samples with at least 2364 characters)
def get_lowest_chars_samples_quant(human_samples_file, ai_samples_file, alphabet):
    
    lowest_sample_size = LOWEST_SAMPLE_SIZE
    
    lowest_human = np.inf
    count_human = 0
    with open(human_samples_file, "r") as f:
        for line in f:
            # remove special characters
            line_regex = re.sub(f"[^{alphabet}]", "", line.lower())
            
            if len(line_regex) >= lowest_sample_size:
                count_human += 1
            
            if len(line_regex) >= lowest_sample_size and len(line_regex) < lowest_human:
                lowest_human = len(line_regex)

    lowest_ai = np.inf
    count_ai = 0
    with open(ai_samples_file, "r") as f:
        for line in f:
            # remove special characters
            line_regex = re.sub(f"[^{alphabet}]", "", line.lower())
            
            if len(line_regex) >= lowest_sample_size:
                count_ai += 1
                
            if len(line_regex) >= lowest_sample_size and len(line_regex) < lowest_ai:
                lowest_ai = len(line_regex)
                
    print("Lowest human Sample size:", lowest_human)
    print("Lowest AI Sample size:", lowest_ai)
    
    print("Number of human samples with at least 1000 characters:", count_human)
    print("Number of AI samples with at least 1000 characters:", count_ai)
    
    return min(lowest_human, lowest_ai)
    

def get_lowest_chars_samples(human_samples_file, ai_samples_file, alphabet):
    
    lowest_sample_size = LOWEST_SAMPLE_SIZE
    
    human_samples = []
    with open(human_samples_file, "r") as f:
        for line in f:
            # remove special characters
            line_regex = re.sub(f"[^{alphabet}]", "", line.lower())
            
            if len(line_regex) >= lowest_sample_size:
                human_samples.append(line_regex)

    ai_samples = []
    with open(ai_samples_file, "r") as f:
        for line in f:
            # remove special characters
            line_regex = re.sub(f"[^{alphabet}]", "", line.lower())
            
            if len(line_regex) >= lowest_sample_size:
                ai_samples.append(line_regex)
    
    human_samples = sorted(human_samples, key=lambda x: len(x))
    ai_samples = sorted(ai_samples, key=lambda x: len(x))
    
    if len(human_samples) > MAX_NUMBER_OF_SAMPLES:
        human_samples = human_samples[:MAX_NUMBER_OF_SAMPLES]
    if len(ai_samples) > MAX_NUMBER_OF_SAMPLES:
        ai_samples = ai_samples[:MAX_NUMBER_OF_SAMPLES]
        
    if len(human_samples) > len(ai_samples):
        # order human samples by size
        human_samples = human_samples[:len(ai_samples)]
    elif len(ai_samples) > len(human_samples):
        # order ai samples by size
        ai_samples = ai_samples[:len(human_samples)]
    
    print("Number of human samples:", len(human_samples))
    print("Number of AI samples:", len(ai_samples))
    
    return human_samples, ai_samples


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
    
    with open(args.alphabet, "r") as f:
        alphabet = f.read().strip()
    print("alphabet:", alphabet)
    print("type(alphabet):", type(alphabet))
    
    lowest_sample_size = get_lowest_chars_samples_quant(args.not_gpt, args.gpt, alphabet)
    
    print("Lowest sample size:", lowest_sample_size)
    
    human_samples, ai_samples = get_lowest_chars_samples(args.not_gpt, args.gpt, alphabet)
    # with this arguments and a value of 2364 for the lowest sample size, we get 5010 but only used 5000 for both human and AI samples using the dataset 1
    
    # Create the folder that will contain the temporary datasets
    Path("data/temp/target").mkdir(parents=True, exist_ok=True)
    
    
    accs = []
    human_accs = []
    ai_accs = []
    samples = []
    for i in range(STEP, lowest_sample_size, STEP):
        print(f"\nSamples 0 --> {i} chars")
        
        with open("data/temp/target/human_test.txt", "w") as f:
            for line in human_samples:
                f.write(line[:i] + "\n")
                    
        with open("data/temp/target/ai_test.txt", "w") as f:
            for line in ai_samples:
                f.write(line[:i] + "\n")
        
        print("Evaluating model with human samples")
        samples_human, correct_human = run_was_chatted(f"models/k_{K}", "data/temp/target/human_test.txt", ALPHA, "human")

        print("Evaluating model with AI samples")
        samples_ai, correct_ai = run_was_chatted(f"models/k_{K}", "data/temp/target/ai_test.txt", ALPHA, "ai")

        acc = (correct_human + correct_ai) / (samples_human + samples_ai)
        human_acc = correct_human / samples_human
        ai_acc = correct_ai / samples_ai
        
        accs.append(acc)
        human_accs.append(human_acc)
        ai_accs.append(ai_acc)
        samples.append(i)
        
        print(f"Accuracy: {acc}")
    
    
    os.system("rm -r data/temp")
    
    # plot Accuracy vs Size of samples
    plt.figure(figsize=(10, 6))
    plt.plot(samples, accs)
    plt.xlabel("Size of samples")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Size of samples")
    plt.savefig("src/other/target_accs.png")
    
    # clear the plot
    plt.clf()
    
    # plot Human Accuracy vs AI Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(samples, human_accs, label="Human", color="red")
    plt.plot(samples, ai_accs, label="AI", color="blue")
    plt.xlabel("Size of samples")
    plt.ylabel("Accuracy")
    plt.title("Human Accuracy vs AI Accuracy")
    plt.legend()
    plt.savefig("src/other/target_human_ai_accs.png")
    
    
    
        
    
    
    