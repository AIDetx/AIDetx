# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "andythetechnerd03/BERT-tiny_AI-Human"
human_dataset = "data/data1/human_val.txt"
ai_dataset = "data/data1/ai_val.txt"

# model_id = "0xnu/AGTD-v0.1"
# human_dataset = "data/data2/human_val.txt"
# ai_dataset = "data/data2/ai_val.txt"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512
model = AutoModelForSequenceClassification.from_pretrained(model_id)


human_correct = 0
human_incorrect = 0
ai_correct = 0
ai_incorrect = 0
for type, dataset in enumerate([human_dataset, ai_dataset]):
    print(f"Validating {dataset}...")
    
    f = open(dataset, "r")
    
    # read the file line by line
    for line in f:
        # Input text
        text = line.strip()
        
        human_prob_sum = 0
        ai_prob_sum = 0
        
        # print(f"Text: {text}")
        
        for j in range(0, len(text), 512):
            mini_text = text[j:j+512]

            # Preprocess the text
            inputs = tokenizer(mini_text, return_tensors='pt')

            # Run the model
            outputs = model(**inputs)

            # Interpret the output
            logits = outputs.logits

            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Assuming the first class is 'human' and the second class is 'ai'
            human_prob, ai_prob = probabilities.detach().numpy()[0]
            
            human_prob_sum += human_prob
            ai_prob_sum += ai_prob

        # Determine if the text is human or AI-generated
        if human_prob_sum > ai_prob_sum:
            if type == 0:
                human_correct += 1
            else:
                ai_incorrect += 1
        else:
            if type == 1:
                ai_correct += 1
            else:
                human_incorrect += 1

    f.close()

print(f"Human samples: {human_correct} correct, {human_incorrect} incorrect")
print(f"AI samples: {ai_correct} correct, {ai_incorrect} incorrect")

# Accuracy
total_samples = human_correct + human_incorrect + ai_correct + ai_incorrect
accuracy = (human_correct + ai_correct) / total_samples
print(f"Accuracy: {accuracy}")

# F1 score
human_precision = human_correct / (human_correct + ai_incorrect)
human_recall = human_correct / (human_correct + human_incorrect)
human_f1 = 2 * human_precision * human_recall / (human_precision + human_recall)
print(f"Human F1 score: {human_f1}")

ai_precision = ai_correct / (ai_correct + human_incorrect)
ai_recall = ai_correct / (ai_correct + ai_incorrect)
ai_f1 = 2 * ai_precision * ai_recall / (ai_precision + ai_recall)
print(f"AI F1 score: {ai_f1}")

f1 = (human_f1 + ai_f1) / 2
print(f"F1 score: {f1}")