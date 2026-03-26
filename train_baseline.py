import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ocr_utils import OCRDataset, CRNN, DEVICE, char_to_idx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_csv = os.path.join(BASE_DIR, "data", "processed", "train.csv")
val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")
IMG_HEIGHT = 32
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 50
MAX_TEXT_LEN = 32

# Custom collate
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths

# Data
train_dataset = OCRDataset(train_csv, base_dir=BASE_DIR, max_len=MAX_TEXT_LEN)
val_dataset = OCRDataset(val_csv, base_dir=BASE_DIR, max_len=MAX_TEXT_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Model init
model = CRNN(
    num_classes=len(char_to_idx) + 1,
    hidden_size=512
).to(DEVICE)

criterion = nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=4.33e-4)

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

torch.save(model.state_dict(), os.path.join(BASE_DIR, "crnn_baseline.pth"))
print("MODEL SAVED to crnn_baseline.pth")