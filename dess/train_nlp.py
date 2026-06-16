import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

# =========================
# CONFIG
# =========================
# MODEL_TYPE = "flan"
MODEL_TYPE = "bart"
CSV_PATH = "dataset/dataset.csv"
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-5

# =========================
# DATASET
# =========================
class NLPDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------------
        # IMPORTANT FIX: TASK PROMPT
        # -------------------------
        input_text = f"convert to math expression: {row['text']}"

        target_text = str(row["target"])

        inputs = self.tokenizer(
            input_text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer(
            target_text,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }

# =========================
# MODEL
# =========================
if MODEL_TYPE == "flan":
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

elif MODEL_TYPE == "bart":
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

else:
    raise ValueError("MODEL_TYPE must be 'flan' or 'bart'")

# =========================
# DATA
# =========================
dataset = NLPDataset(CSV_PATH, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss={total_loss/len(loader):.4f}")

# =========================
# SAVE
# =========================
torch.save(model.state_dict(), f"{MODEL_TYPE}_nlp.pt")
print("Saved:", f"{MODEL_TYPE}_nlp.pt")