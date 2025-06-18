from analysis import *
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

model = SentimentAnalyzer()
model.load_state_dict(torch.load("sentiment_model_0.pt"))
model.to(device)
model.eval()

# dummy dataset to use tokenizer function
train_dataset = SentimentDataset(length=1)

def getConfidence(tens):
    prob = round(float(tens) * 100.0)
    if prob < 50.0:
        return 100.0 - prob
    else:
        return prob
while(True):
    test_str = input("Enter a sentence to evaluate: ")
    test_tokenized = train_dataset.tokenize_string(test_str)
    with torch.no_grad():
        input_id, attention_mask = test_tokenized["input_ids"], test_tokenized["attention_mask"]
        preds = model(input_id.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device))
        print("Sentiment of sentence is: ", "Positive" if preds[0] > 0.5 else "Negative", f"| Confidence: {getConfidence(preds[0])}%")
