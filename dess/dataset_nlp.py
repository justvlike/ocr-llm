import pandas as pd
import torch
from torch.utils.data import Dataset


class NLPDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_in=64, max_out=32):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        inp = str(row["text"])
        tgt = str(row["target"])

        model_in = self.tokenizer(
            inp,
            max_length=self.max_in,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        model_out = self.tokenizer(
            tgt,
            max_length=self.max_out,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = model_out["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_in["input_ids"].squeeze(),
            "attention_mask": model_in["attention_mask"].squeeze(),
            "labels": labels,
        }