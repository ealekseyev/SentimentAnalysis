from analysis import *
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

train_dataset = SentimentDataset(length=25000)
test_dataset = SentimentDataset(train=False, length=500)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

model = SentimentAnalyzer()
model.load_state_dict(torch.load("sentiment_model_0.pt"))
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#test string

NUM_EPOCHS = 10
for i in range(NUM_EPOCHS):
    # train
    model.train()
    for batch, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
        labels = labels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Batch {batch} loss: {loss.item()}")
    torch.save(model.state_dict(), f"sentiment_model_{i}.pt")
    # test model
    model.eval()
    test_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        input_ids, attention_mask, labels = next(iter(test_dataloader))
        preds = model(input_ids.to(device), attention_mask.to(device))
        test_loss = criterion(preds, labels.to(device))
        # compute accuracy
        preds = (preds.cpu() > 0.5).float()
        correct_preds += (preds.cpu() == labels.cpu()).sum().item()
        total_preds += labels.cpu().size(0)
    print(f"Epoch {i} test loss: {test_loss.item()} | Accuracy: {correct_preds / total_preds:.3f}")
