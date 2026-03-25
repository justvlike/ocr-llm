import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from ocr_utils import OCRDataset, DEVICE, char_to_idx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_csv = os.path.join(BASE_DIR, "data", "processed", "train.csv")
val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")

IMG_HEIGHT = 32
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 70
MAX_TEXT_LEN = 32

# Augs
train_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),

    # Rotation
    A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.3),

    # Diff noise types
    A.OneOf([
        A.GaussNoise(var_limit=(5.0, 30.0)),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        A.CoarseDropout(max_holes=3, max_height=4, max_width=4)
    ], p=0.4),

    # Blur
    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
        A.GaussianBlur(blur_limit=3)
    ], p=0.3),

    # Brightness and contrast
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

    # Elastic deformation
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=5, p=0.2),

    # Normalization
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# Custom collate
def collate_fn(batch):
    images, labels, lengths = zip(*batch)

    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return images, labels, lengths

# Dataset
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Model
class CRNNImproved(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.rnn = nn.LSTM(
            input_size=256 * (IMG_HEIGHT // 4),
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)

        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(b, w, c * h)

        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)

        x = x.permute(1, 0, 2)

        return x

if __name__ == "__main__":

    # Init model
    model = CRNNImproved(num_classes=len(char_to_idx) + 1).to(DEVICE)

    criterion = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Train loop
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

    torch.save(model.state_dict(), os.path.join(BASE_DIR, "crnn_improved.pth"))
    print("MODEL SAVED to crnn_improved.pth")