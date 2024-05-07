import os
import subprocess
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

TRAIN = "./bin/train"
WAS_CHATTED = "./bin/was_chatted"
ALPHA = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='dataset to use', default='data/data1')
parser.add_argument('-a', '--alphabet', help='file with the alphabet', default='data/alphabet.txt')
parser.add_argument('-k', help='context length', default=8)
parser.add_argument('-n', help='number of sample points for analysis', default=20, type=int)
parser.add_argument('-s', '--step', help='number of chars to add in each iter', default=100_000, type=int)
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

    # create temp folder and empty files
    Path(f"{args.dataset}/temp").mkdir(parents=True, exist_ok=True)
    with open(f"{args.dataset}/temp/human_train.txt", "w") as f: pass
    with open(f"{args.dataset}/temp/ai_train.txt", "w") as f: pass

    h_file = open(f"{args.dataset}/human_train.txt", "r")
    g_file = open(f"{args.dataset}/ai_train.txt", "r")

    samples = []
    accs = []
    f1s = []

    for i in range(args.n):
        print(f"\n====== Iter: {i} ======")

        total_chars = 0
        with open(f"{args.dataset}/temp/human_train.txt", "a") as f:
            while total_chars < args.step:
                line = h_file.readline()
                total_chars += len(line)
                f.write(line)

        total_chars = 0
        with open(f"{args.dataset}/temp/ai_train.txt", "a") as f:
            while total_chars < args.step:
                line = g_file.readline()
                total_chars += len(line)
                f.write(line)

        Path(f"models/references/{args.dataset.split('/')[1]}/{i}").mkdir(parents=True, exist_ok=True)

        print("Training model")
        run_train(f"{args.dataset}/temp/human_train.txt", f"{args.dataset}/temp/ai_train.txt", f"models/references/{args.dataset.split('/')[1]}/{i}", args.alphabet, args.k)

        print("Evaluating model with human samples")
        samples_human, correct_human = run_was_chatted(f"models/references/{args.dataset.split('/')[1]}/{i}", f"{args.dataset}/human_test.txt", ALPHA, "human")

        print("Evaluating model with AI samples")
        samples_ai, correct_ai = run_was_chatted(f"models/references/{args.dataset.split('/')[1]}/{i}", f"{args.dataset}/ai_test.txt", ALPHA, "ai")


        acc = (correct_human + correct_ai) / (samples_human + samples_ai)
        accs.append(acc)

        # ai is the positive class and human is the negative class
        y_true = [0] * samples_human + [1] * samples_ai
        y_pred = [0] * correct_human
        y_pred += [1] * (samples_human - correct_human)
        y_pred += [1] * correct_ai
        y_pred += [0] * (samples_ai - correct_ai)
        
        f1 = f1_score(y_true, y_pred)
        print(f"F1 Score: {f1}")
        f1s.append(f1)

        samples.append((i+1) * args.step)
        print(f"Accuracy: {acc}")

    h_file.close()
    g_file.close()
    os.system(f"rm -r {args.dataset}/temp")
    os.system("rm -r models/references")

    # save results as csv
    with open(f"src/other/references_{args.dataset.split('/')[1]}.csv", "w") as f:
        f.write("Samples,Accuracy,F1\n")
        for i in range(len(accs)):
            f.write(f"{samples[i]},{accs[i]}, {f1s[i]}\n")
    
    # plot accs
    plt.plot(samples, accs)
    plt.xlabel("Overall size of the reference texts")
    plt.ylabel("Accuracy")
    plt.savefig(f"src/other/references_{args.dataset.split('/')[1]}.png", dpi=600)
    plt.show()

    # plot f1s
    plt.plot(samples, f1s)
    plt.xlabel("Overall size of the reference texts")
    plt.ylabel("F1 Score")
    plt.savefig(f"src/other/references_{args.dataset.split('/')[1]}_f1.png", dpi=600)
    plt.show()
