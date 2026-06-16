import os
import random
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ocr_utils import (
    DEVICE,
    char_to_idx,
    idx_to_char,
    cer,
    wer
)

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IAM_DIR = os.path.join(BASE_DIR, "data", "IAM", "raw")
EMNIST_DIR = os.path.join(BASE_DIR, "data", "EMNIST", "raw")

MODEL_PATH = os.path.join(BASE_DIR, "crnn_transformer_aug_0.pth")

BATCH_SIZE = 8
MAX_SAMPLES = 3000

USE_IAM = True
USE_EMNIST = True


# =========================
# IAM
# =========================
def load_iam():
    ascii_file = os.path.join(IAM_DIR, "ascii", "lines.txt")
    lines_dir = os.path.join(IAM_DIR, "lines")

    samples = []

    with open(ascii_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9 or parts[1] != "ok":
                continue

            img_id = parts[0]
            text = " ".join(parts[8:])

            folder = img_id.split("-")[0]
            sub = f"{folder}-{img_id.split('-')[1]}"

            path = os.path.join(lines_dir, folder, sub, f"{img_id}.png")

            if os.path.exists(path):
                samples.append((path, text))

    return samples[:MAX_SAMPLES]


# =========================
# EMNIST (FIXED)
# =========================
def load_emnist():
    def read_images(p):
        with open(p, "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, r, c)

    def read_labels(p):
        with open(p, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    img_p = os.path.join(EMNIST_DIR, "emnist-balanced-test-images-idx3-ubyte")
    lbl_p = os.path.join(EMNIST_DIR, "emnist-balanced-test-labels-idx1-ubyte")

    imgs = read_images(img_p)
    labels = read_labels(lbl_p)

    samples = []

    for i in range(min(MAX_SAMPLES, len(imgs))):
        img = np.transpose(imgs[i])
        img = 255 - img

        img = Image.fromarray(img).convert("L")

        # FIX: real stable mapping (not char_to_idx hack)
        char_set = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        text = char_set[labels[i] % len(char_set)]

        samples.append((img, text))

    return samples


# =========================
# DATASET
# =========================
class EvalDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, text = self.samples[idx]

        if isinstance(img, str):
            img = Image.open(img).convert("L")

        img = img.resize((256, 32))
        img = np.array(img).astype(np.float32) / 255.0

        img = torch.tensor(img).unsqueeze(0)

        return img, text


# =========================
# MODEL
# =========================
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )

        self.fc = nn.Linear(256, len(char_to_idx) + 3)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape

        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x = x[:, :, :256]

        x = self.transformer(x)
        return self.fc(x).log_softmax(2)


# =========================
# CTC decode
# =========================
def decode(logits):
    out = logits.argmax(2)

    res = []

    for seq in out:
        prev = -1
        txt = []

        for i in seq:
            i = i.item()

            if i != prev and i != 0:
                txt.append(idx_to_char.get(i, ""))

            prev = i

        res.append("".join(txt))

    return res


# =========================
# MAIN
# =========================
def main():

    print("Loading data...")

    samples = []

    if USE_IAM:
        samples += load_iam()

    if USE_EMNIST:
        samples += load_emnist()

    random.shuffle(samples)

    dataset = EvalDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    print("Loading model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_cer, all_wer = [], []

    print("Evaluating...")

    with torch.no_grad():
        for imgs, texts in loader:
            imgs = imgs.to(DEVICE)

            logits = model(imgs)
            preds = decode(logits)

            for p, t in zip(preds, texts):
                all_cer.append(cer(p, t))
                all_wer.append(wer(p, t))

    print("\n======================")
    print(f"CER: {sum(all_cer)/len(all_cer):.4f}")
    print(f"WER: {sum(all_wer)/len(all_wer):.4f}")
    print("======================\n")

    print("VISUAL CHECK\n")

    for i in random.sample(range(len(dataset)), 5):
        img, true = dataset[i]

        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(DEVICE))
            pred = decode(logits)[0]

        print("-" * 40)
        print("T:", true)
        print("P:", pred)


if __name__ == "__main__":
    main()