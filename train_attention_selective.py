import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from ocr_utils import OCRDataset, DEVICE, char_to_idx

# =========================================
# CONFIG
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_csv = os.path.join(BASE_DIR, "data", "processed", "train.csv")
val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")

IMG_HEIGHT = 32
IMG_WIDTH = 256

BATCH_SIZE = 16
EPOCHS = 50

MAX_TEXT_LEN = 16

HIDDEN_SIZE = 256
EMBED_SIZE = 128

SOS_TOKEN = len(char_to_idx) + 1
EOS_TOKEN = len(char_to_idx) + 2
NUM_CLASSES = len(char_to_idx) + 3

TEACHER_FORCING_RATIO = 0.85  # slightly lower = less looping

# =========================================
# AUGMENT MODES
# =========================================
# 0 = baseline
# 1 = noise
# 2 = blur
# 3 = rotation
# 4 = brightness/contrast
AUGMENT_MODE = 4


def augment_image(img_tensor):
    img = img_tensor.squeeze().numpy()
    img = (img * 255).astype(np.uint8)

    if AUGMENT_MODE == 1:
        noise = np.random.normal(0, 15, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif AUGMENT_MODE == 2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    elif AUGMENT_MODE == 3:
        angle = random.uniform(-3, 3)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    elif AUGMENT_MODE == 4:
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)


class AugmentedOCRDataset(Dataset):
    def __init__(self, csv_path, base_dir, max_len):
        self.dataset = OCRDataset(csv_path, base_dir=base_dir, max_len=max_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, length = self.dataset[idx]
        image = augment_image(image)
        return image, label, length


def collate_fn(batch):
    images, labels, lengths = zip(*batch)

    images = torch.stack(images)

    padded = torch.full((len(labels), MAX_TEXT_LEN), EOS_TOKEN, dtype=torch.long)

    for i, lab in enumerate(labels):
        l = min(len(lab), MAX_TEXT_LEN - 1)
        padded[i, 0] = SOS_TOKEN
        padded[i, 1:l + 1] = lab[:l]

    return images, padded, torch.tensor(lengths)


# =========================================
# ATTENTION
# =========================================
class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, hidden, enc):
        seq = enc.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq, 1)

        energy = torch.tanh(self.attn(torch.cat([hidden, enc], dim=2)))
        attn = self.v(energy).squeeze(2)

        return torch.softmax(attn, dim=1)


class AttentionDecoder(nn.Module):
    def __init__(self, out_size, hidden, emb):
        super().__init__()

        self.embedding = nn.Embedding(out_size, emb)
        self.dropout = nn.Dropout(0.2)

        self.attention = Attention(hidden)

        self.rnn = nn.LSTM(hidden + emb, hidden, batch_first=True)

        self.fc = nn.Linear(hidden * 2, out_size)

    def forward(self, x, h, c, enc):
        x = self.embedding(x.unsqueeze(1))
        x = self.dropout(x)

        attn = self.attention(h[-1], enc).unsqueeze(1)

        context = torch.bmm(attn, enc)

        rnn_in = torch.cat([x, context], dim=2)

        out, (h, c) = self.rnn(rnn_in, (h, c))

        out = out.squeeze(1)
        context = context.squeeze(1)

        return self.fc(torch.cat([out, context], dim=1)), h, c


# =========================================
# MODEL
# =========================================
class CRNNAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )

        self.encoder = nn.LSTM(
            256 * (IMG_HEIGHT // 4),
            HIDDEN_SIZE,
            bidirectional=True,
            batch_first=True
        )

        self.reduce = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)

        self.decoder = AttentionDecoder(num_classes, HIDDEN_SIZE, EMBED_SIZE)

    def forward(self, images, targets=None, teacher_forcing=0.5):

        x = self.cnn(images)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)

        enc, (h0, c0) = self.encoder(x)

        enc = self.reduce(enc)

        hidden = torch.tanh(self.reduce(torch.cat([h0[0], h0[1]], dim=1))).unsqueeze(0)
        cell = torch.zeros_like(hidden)

        out = torch.zeros(b, MAX_TEXT_LEN, NUM_CLASSES).to(images.device)

        inp = torch.full((b,), SOS_TOKEN, device=images.device)

        for t in range(MAX_TEXT_LEN):

            pred, hidden, cell = self.decoder(inp, hidden, cell, enc)

            out[:, t] = pred

            best = pred.argmax(1)

            # SAFER STOP (per-sample ignored, but prevents runaway)
            if t > 5 and (best == EOS_TOKEN).float().mean() > 0.7:
                break

            if targets is not None and random.random() < teacher_forcing:
                inp = targets[:, t]
            else:
                inp = best

        return out


# =========================================
# TRAIN
# =========================================
model = CRNNAttention(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=EOS_TOKEN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_ds = AugmentedOCRDataset(train_csv, BASE_DIR, MAX_TEXT_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

for epoch in range(EPOCHS):
    model.train()
    total = 0

    for img, lab, _ in train_loader:
        img, lab = img.to(DEVICE), lab.to(DEVICE)

        out = model(img, lab)

        loss = criterion(out.reshape(-1, NUM_CLASSES), lab.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total += loss.item()

    print(f"Epoch {epoch+1}: {total:.4f}")


torch.save(model.state_dict(), os.path.join(BASE_DIR, f"crnn_attention_aug_{AUGMENT_MODE}.pth"))

print("DONE")