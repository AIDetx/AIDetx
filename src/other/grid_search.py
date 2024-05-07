import subprocess
from pathlib import Path
import pandas as pd

OUTPUT = "src/other/grid_search.csv"
DATASETS = ["data1", "data2"]
K_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
ALPHAS = [0.1, 0.5, 1, 5, 10]

colls = ["dataset", "file", "time", "k", "alpha", "samples", "hits", "misses"]
df = pd.DataFrame(columns=colls)

def model_cmd(dataset, k):
    return f"./bin/train -h data/{dataset}/human_train.txt -g data/{dataset}/ai_train.txt -k {k} -o models/grid_search/{dataset}_{k} -a data/alphabet.txt"

def validate_cmd(dataset, k, alpha, type_):
    return f"./bin/was_chatted -m models/grid_search/{dataset}_{k} -a {alpha} -d data/{dataset}/{type_}_val.txt"

# create models
for dataset in DATASETS:
    for k in K_VALUES:
        if Path(f"models/grid_search/{dataset}_{k}").exists():
            continue
        cmd = model_cmd(dataset, k)
        print(cmd)
        subprocess.run(cmd, shell=True)

# validate models
i = 0
for dataset in DATASETS:
    for k in K_VALUES:
        for alpha in ALPHAS:
            for type_ in ["human", "ai"]:
                cmd = validate_cmd(dataset, k, alpha, type_)
                print(cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True)
                print(result.stdout.decode("utf-8"))
                result = result.stdout.decode("utf-8").split("\n")[:-1]
                n_samples = int(result[-5].split(":")[1])
                human_hits = int(result[-4].split(":")[1].split(",")[0])
                time = float(result[-1].split(":")[1])
                hits = human_hits if type_ == "human" else n_samples - human_hits
                misses = n_samples - hits
                df.loc[i] = [dataset, type_, time, k, alpha, n_samples, hits, misses]
                i += 1

df.to_csv(OUTPUT, index=False)



