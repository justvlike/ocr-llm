import os
import random
import struct
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from tqdm import tqdm

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

BATCH_SIZE = 16
EPOCHS = 25
LR = 2e-4

MAX_TEXT_LEN = 32
IMG_H, IMG_W = 32, 256

HIDDEN = 256

SOS = len(char_to_idx) + 1
EOS = len(char_to_idx) + 2
NUM_CLASSES = len(char_to_idx) + 3

# =========================
# AUGMENT MODES
# =========================
# 0 = baseline
# 1 = noise
# 2 = blur
# 3 = rotation
# 4 = brightness/contrast

AUGMENT_MODE = 0

# =========================
# EMNIST (NO DISK!)
# =========================
def read_idx_images(path):
    with open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, r, c)


def read_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_emnist():
    img_p = os.path.join(EMNIST_DIR, "emnist-balanced-train-images-idx3-ubyte")
    lbl_p = os.path.join(EMNIST_DIR, "emnist-balanced-train-labels-idx1-ubyte")

    imgs = read_idx_images(img_p)
    labels = read_idx_labels(lbl_p)

    chars = list(char_to_idx.keys())

    samples = []

    for i in range(min(5000, len(imgs))):
        img = np.transpose(imgs[i])
        img = 255 - img

        img = Image.fromarray(img).convert("L")
        img = img.resize((IMG_W, IMG_H))

        arr = np.array(img) / 255.0
        tensor = torch.tensor(arr).unsqueeze(0).float()

        text = chars[labels[i] % len(chars)]

        samples.append((tensor, text))

    return samples


# =========================
# IAM
# =========================
def load_iam():
    ascii_file = os.path.join(IAM_DIR, "ascii", "lines.txt")
    lines_dir = os.path.join(IAM_DIR, "lines")

    samples = []

    with open(ascii_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
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
                img = Image.open(path).convert("L").resize((IMG_W, IMG_H))
                arr = np.array(img) / 255.0
                tensor = torch.tensor(arr).unsqueeze(0).float()

                samples.append((tensor, text))

    return samples[:5000]


# =========================
# AUGMENTATION
# =========================
def augment(x):
    img = (x.squeeze().numpy() * 255).astype(np.uint8)

    if AUGMENT_MODE == 1:
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255)

    elif AUGMENT_MODE == 2:
        img = Image.fromarray(img).filter(ImageFilter.GaussianBlur(1))
        img = np.array(img)

    elif AUGMENT_MODE == 3:
        img = Image.fromarray(img)
        img = img.rotate(random.uniform(-3, 3), fillcolor=255)
        img = np.array(img)

    elif AUGMENT_MODE == 4:

        alpha = random.uniform(0.8, 1.2)

        beta = random.randint(-20, 20)

        img = np.clip(
            img.astype(np.float32) * alpha + beta,
            0,
            255
        ).astype(np.uint8)

    img = img / 255.0

    return torch.tensor(img).unsqueeze(0).float()


# =========================
# DATASET
# =========================
class OCRDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, text = self.samples[i]
        img = augment(img)

        label = [char_to_idx.get(c, 0) for c in text[:MAX_TEXT_LEN]]

        return img, torch.tensor(label), len(label)


def collate(batch):
    imgs, labels, lens = zip(*batch)

    imgs = torch.stack(imgs)

    padded = torch.full((len(labels), MAX_TEXT_LEN), EOS, dtype=torch.long)

    for i, l in enumerate(labels):
        padded[i, :len(l)] = l

    return imgs, padded, torch.tensor(lens)


# =========================
# MODEL
# =========================
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

        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape

        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x = x[:, :, :256]

        x = self.transformer(x)
        return self.fc(x).log_softmax(2)


# =========================
# TRAIN
# =========================
model = Model().to(DEVICE)

ctc = nn.CTCLoss(blank=0, zero_infinity=True)
opt = torch.optim.Adam(model.parameters(), lr=LR)

train = OCRDataset(load_iam() + load_emnist())
loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

for epoch in range(EPOCHS):
    model.train()
    total = 0

    for img, label, _ in tqdm(loader):
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        out = model(img)

        input_len = torch.full((img.size(0),), out.size(1), dtype=torch.long)
        target_len = torch.tensor([len(l[l != EOS]) for l in label])

        loss = ctc(
            out.permute(1, 0, 2),
            label,
            input_len,
            target_len
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch+1}: {total:.4f}")


torch.save(model.state_dict(), os.path.join(BASE_DIR, f"crnn_transformer_aug_{AUGMENT_MODE}.pth"))

print("DONE")