import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ocr_utils import OCRDataset, CRNN, DEVICE, char_to_idx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_csv = os.path.join(BASE_DIR, "data", "processed", "train.csv")
val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")

IMG_HEIGHT = 32
IMG_WIDTH = 256

BATCH_SIZE = 16
EPOCHS = 50
MAX_TEXT_LEN = 32

# =========================
# AUGMENT MODES
# =========================
# 0 = exact baseline
# 1 = baseline + noise
# 2 = baseline + blur
# 3 = baseline + rotation
# 4 = baseline + contrast/brightness

AUGMENT_MODE = 4

augmentations = {
    # =========================
    # EXACT BASELINE
    # =========================
    0: None,

    # =========================
    # BASELINE + NOISE
    # =========================
    1: A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),

        A.GaussNoise(
            std_range=(0.01, 0.03),
            p=0.25
        ),

        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]),

    # =========================
    # BASELINE + BLUR
    # =========================
    2: A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),

        A.GaussianBlur(
            blur_limit=(3, 3),
            p=0.20
        ),

        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]),

    # =========================
    # BASELINE + ROTATION
    # =========================
    3: A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),

        A.Rotate(
            limit=3,
            p=0.20
        ),

        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]),
    # =========================
    # BASELINE + BRIGHTNESS / CONTRAST
    # =========================
    4: A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),

        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.25
        ),

        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
}

train_transform = augmentations[AUGMENT_MODE]

print(f"USING AUGMENT MODE: {AUGMENT_MODE}")

# =========================
# COLLATE
# =========================
def collate_fn(batch):
    images, labels, lengths = zip(*batch)

    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return images, labels, lengths

# =========================
# DATA
# =========================
train_dataset = OCRDataset(
    train_csv,
    base_dir=BASE_DIR,
    max_len=MAX_TEXT_LEN,
    transform=train_transform
)

val_dataset = OCRDataset(
    val_csv,
    base_dir=BASE_DIR,
    max_len=MAX_TEXT_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# =========================
# MODEL INIT
# =========================
model = CRNN(
    num_classes=len(char_to_idx) + 1,
    hidden_size=512
).to(DEVICE)

criterion = nn.CTCLoss(blank=0)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=4.33e-4
)

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for images, labels, label_lengths in train_loader:
        images = images.to(DEVICE)

        labels = labels.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)

        outputs = model(images)

        input_lengths = torch.full(
            (images.size(0),),
            outputs.size(0),
            dtype=torch.long
        ).to(DEVICE)

        loss = criterion(
            outputs.log_softmax(2),
            labels,
            input_lengths,
            label_lengths
        )

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# SAVE
# =========================
save_name = f"crnn_selective_aug_{AUGMENT_MODE}.pth"

torch.save(
    model.state_dict(),
    os.path.join(BASE_DIR, save_name)
)

print(f"MODEL SAVED: {save_name}")