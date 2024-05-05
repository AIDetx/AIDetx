import os
import subprocess
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

TRAIN = "./bin/train"
WAS_CHATTED = "./bin/was_chatted"
ALPHA = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('-ng', '--not_gpt', help='file with human text samples', default='data/human_train.txt')
parser.add_argument('-g', '--gpt', help='file with gpt text samples', default='data/ai_train.txt')
parser.add_argument('-a', '--alphabet', help='file with the alphabet', default='data/alphabet.txt')
parser.add_argument('-k', help='context length', default=5)
parser.add_argument('-n', help='number of sample points for analysis', default=20, type=int)
parser.add_argument('-m', help='max number of samples for analysis', default=5000, type=int)
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

    h_samples = 0
    g_samples = 0

    with open(args.not_gpt, "r") as f:
        for line in f:
            h_samples += 1

    with open(args.gpt, "r") as f:
        for line in f:
            g_samples += 1
    
    max_samples = min(h_samples, g_samples)
    if args.m > max_samples:
        print(f"ERROR: Max number of samples for analysis is {max_samples}")
        exit(0)

    step = args.m // args.n

    # create temp folder and empty files
    Path("data/temp").mkdir(parents=True, exist_ok=True)
    with open("data/temp/human_train.txt", "w") as f: pass
    with open("data/temp/ai_train.txt", "w") as f: pass

    h_file = open(args.not_gpt, "r")
    g_file = open(args.gpt, "r")

    accs = []
    samples = []

    for i in range(args.n):
        print(f"\n====== Samples 0 --> {(i+1)*step} ======")

        with open("data/temp/human_train.txt", "a") as f:
            for j in range(step):
                f.write(h_file.readline())

        with open("data/temp/ai_train.txt", "a") as f:
            for j in range(step):
                f.write(g_file.readline())

        Path(f"models/references/{i}").mkdir(parents=True, exist_ok=True)

        print("Training model")
        run_train("data/temp/human_train.txt", "data/temp/ai_train.txt", f"models/references/{i}", args.alphabet, args.k)

        print("Evaluating model with human samples")
        samples_human, correct_human = run_was_chatted(f"models/references/{i}", "data/human_test.txt", ALPHA, "human")

        print("Evaluating model with AI samples")
        samples_ai, correct_ai = run_was_chatted(f"models/references/{i}", "data/ai_test.txt", ALPHA, "ai")

        acc = (correct_human + correct_ai) / (samples_human + samples_ai)
        accs.append(acc)
        samples.append((i+1) * step)
        print(f"Accuracy: {acc}")

    h_file.close()
    g_file.close()
    os.system("rm -r data/temp")
    os.system("rm -r models/references")

    # save results as csv
    with open("src/other/references_accs.csv", "w") as f:
        f.write("Samples,Accuracy\n")
        for i in range(len(accs)):
            f.write(f"{samples[i]},{accs[i]}\n")
    
    # plot
    plt.plot(samples, accs)
    plt.xlabel("Number of samples")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of samples")
    plt.savefig("src/other/references_accs.png")
    plt.show()
