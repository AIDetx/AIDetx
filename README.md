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
python3 src/other/generate_data.py
# train a model
./bin/train -h data/data1/human_train.txt -g data/data1/ai_train.txt -o models/data1 -a data/alphabet.txt -k 8
./bin/train -h data/data2/human_train.txt -g data/data2/ai_train.txt -o models/data2 -a data/alphabet.txt -k 8
# test the model
./bin/was_chatted -m models/data1 -d data/data1/human_test.txt -a 0.5
./bin/was_chatted -m models/data1 -d data/data1/ai_test.txt -a 0.5
./bin/was_chatted -m models/data2 -d data/data2/human_test.txt -a 0.5
./bin/was_chatted -m models/data2 -d data/data2/ai_test.txt -a 0.5
```
