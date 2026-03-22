import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import editdistance

# Globals
IMG_HEIGHT = 32
IMG_WIDTH = 256
MAX_TEXT_LEN = 32

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 "
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class OCRDataset(Dataset):
    def __init__(self, csv_path, base_dir=None, max_len=MAX_TEXT_LEN):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_csv(csv_path)

        self.df = self.df.dropna(subset=["text"])
        self.df["text"] = self.df["text"].astype(str)
        self.df = self.df[self.df["text"].str.lower() != "nan"]
        self.df = self.df[self.df["text"].str.strip() != ""]
        self.df = self.df[self.df["text"].str.len() <= max_len]

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor()
        ])

        print(f"Loaded dataset: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def encode(self, text):
        text = str(text).lower()
        return [char_to_idx[c] for c in text if c in char_to_idx]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.base_dir, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(self.encode(row["text"]), dtype=torch.long)

        return img, label, len(label)

# CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.rnn = nn.LSTM(
            input_size=128 * (IMG_HEIGHT // 4),
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # (T, B, C)
        return x

# Greedy decoder
def greedy_decode(output, blank=0):
    arg_maxes = output.argmax(2)  # (T, B)
    arg_maxes = arg_maxes.permute(1, 0)  # (B, T)
    decoded_batch = []

    for seq in arg_maxes:
        prev = blank
        decoded = []
        for idx in seq:
            idx = idx.item()
            if idx != blank and idx != prev:
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx
        decoded_batch.append("".join(decoded))

    return decoded_batch

# Metrics
def cer(pred, target):
    return editdistance.eval(pred, target) / max(len(target), 1)

def wer(pred, target):
    pred_words = pred.split()
    target_words = target.split()
    return editdistance.eval(pred_words, target_words) / max(len(target_words), 1)