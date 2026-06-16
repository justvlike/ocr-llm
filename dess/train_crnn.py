from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from model_crnn import CRNN

# =========================
# PATHS
# =========================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT / "dess/dataset/train/dataset.csv"
VAL_CSV = ROOT / "dess/dataset/test/dataset.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_H = 32
IMG_W = 512

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

# =========================
# charset
# =========================

train_df = pd.read_csv(TRAIN_CSV)

texts = train_df["text"].astype(str)

charset = sorted(set("".join(texts)))

char_to_idx = {c: i + 1 for i, c in enumerate(charset)}

num_classes = len(charset) + 1

# =========================
# dataset
# =========================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])


class DatasetCRNN(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        image_path = str(row["image"])

        image = Image.open(image_path).convert("L")
        image = transform(image)

        text = str(row["text"]).replace("|", " ")

        target = torch.tensor(
            [char_to_idx[c] for c in text if c in char_to_idx],
            dtype=torch.long
        )

        return image, target


def collate_fn(batch):

    images = []
    targets = []
    target_lengths = []

    for img, tgt in batch:
        images.append(img)
        targets.extend(tgt.tolist())
        target_lengths.append(len(tgt))

    images = torch.stack(images)
    targets = torch.tensor(targets)

    target_lengths = torch.tensor(target_lengths)

    return images, targets, target_lengths


# =========================
# loaders
# =========================

train_loader = DataLoader(
    DatasetCRNN(TRAIN_CSV),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    DatasetCRNN(VAL_CSV),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# =========================
# model
# =========================

model = CRNN(num_classes).to(DEVICE)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# train
# =========================

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, targets, target_lengths in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(images)

        logits = logits.log_softmax(2)
        logits = logits.permute(1, 0, 2)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=logits.size(0),
            dtype=torch.long
        ).to(DEVICE)

        loss = criterion(
            logits,
            targets,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()
        loss.backward()

        # 🔥 IMPORTANT STABILITY FIX
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss={total_loss/len(train_loader):.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "charset": charset
        },
        ROOT / "weights/crnn_best.pt"
    )