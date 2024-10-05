# AIDetx

Distinguish between AI-generated text and human-written content using compression techniques.

## Compile

```bash
g++ -Wall -O3 src/train.cpp -o bin/train
g++ -Wall -O3 src/was_chatted.cpp -o bin/was_chatted
```

## Run

```bash
# generate data
python3 src/other/data.py
# train a model
./bin/train -h data/human_train.txt -g data/ai_train.txt -o models/k_5 -a data/alphabet.txt -k 5
# test the model
./bin/was_chatted -m models/k_5 -d data/human_test.txt -a 0.5
./bin/was_chatted -m models/k_5 -d data/ai_test.txt -a 0.5
```
