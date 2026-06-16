import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ocr_utils import OCRDataset, DEVICE, char_to_idx

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

TEACHER_FORCING_RATIO = 0.9

# =========================================
# COLLATE
# =========================================
def collate_fn(batch):
    images, labels, lengths = zip(*batch)

    images = torch.stack(images)

    padded_labels = torch.full(
        (len(labels), MAX_TEXT_LEN),
        EOS_TOKEN,
        dtype=torch.long
    )

    for i, label in enumerate(labels):
        label_len = min(len(label), MAX_TEXT_LEN - 1)

        padded_labels[i, 0] = SOS_TOKEN
        padded_labels[i, 1:label_len + 1] = label[:label_len]

    lengths = torch.tensor(lengths, dtype=torch.long)

    return images, padded_labels, lengths

# =========================================
# DATA
# =========================================
train_dataset = OCRDataset(
    train_csv,
    base_dir=BASE_DIR,
    max_len=MAX_TEXT_LEN
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

# =========================================
# ATTENTION
# =========================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.attn = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

        self.v = nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(
            1,
            seq_len,
            1
        )

        energy = torch.tanh(
            self.attn(
                torch.cat(
                    (hidden, encoder_outputs),
                    dim=2
                )
            )
        )

        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)

# =========================================
# DECODER
# =========================================
class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, embed_size):
        super().__init__()

        self.embedding = nn.Embedding(
            output_size,
            embed_size
        )

        self.attention = Attention(hidden_size)

        self.rnn = nn.LSTM(
            hidden_size + embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(self, input_char, hidden, cell, encoder_outputs):
        input_char = input_char.unsqueeze(1)

        embedded = self.embedding(input_char)

        attention_weights = self.attention(
            hidden[-1],
            encoder_outputs
        )

        attention_weights = attention_weights.unsqueeze(1)

        context = torch.bmm(
            attention_weights,
            encoder_outputs
        )

        rnn_input = torch.cat(
            (embedded, context),
            dim=2
        )

        output, (hidden, cell) = self.rnn(
            rnn_input,
            (hidden, cell)
        )

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc(
            torch.cat(
                (output, context),
                dim=1
            )
        )

        return prediction, hidden, cell

# =========================================
# MODEL
# =========================================
class CRNNAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )

        self.encoder = nn.LSTM(
            input_size=256 * (IMG_HEIGHT // 4),
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.hidden_reduce = nn.Linear(
            HIDDEN_SIZE * 2,
            HIDDEN_SIZE
        )

        self.decoder = AttentionDecoder(
            output_size=num_classes,
            hidden_size=HIDDEN_SIZE,
            embed_size=EMBED_SIZE
        )

    def forward(
        self,
        images,
        targets=None,
        teacher_forcing_ratio=0.5
    ):
        x = self.cnn(images)

        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(
            b,
            w,
            c * h
        )

        encoder_outputs, (hidden, cell) = self.encoder(x)

        encoder_outputs = self.hidden_reduce(
            encoder_outputs
        )

        hidden = torch.tanh(
            self.hidden_reduce(
                torch.cat(
                    (hidden[0], hidden[1]),
                    dim=1
                )
            )
        ).unsqueeze(0)

        cell = torch.zeros_like(hidden)

        batch_size = images.size(0)

        outputs = torch.zeros(
            batch_size,
            MAX_TEXT_LEN,
            NUM_CLASSES
        ).to(DEVICE)

        input_char = torch.full(
            (batch_size,),
            SOS_TOKEN,
            dtype=torch.long
        ).to(DEVICE)

        for t in range(MAX_TEXT_LEN):

            output, hidden, cell = self.decoder(
                input_char,
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:, t] = output

            best_guess = output.argmax(1)

            if (best_guess == EOS_TOKEN).all():
                break

            if (
                targets is not None and
                random.random() < teacher_forcing_ratio
            ):
                input_char = targets[:, t]
            else:
                input_char = best_guess

        return outputs

# =========================================
# INIT
# =========================================
model = CRNNAttention(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    ignore_index=EOS_TOKEN
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =========================================
# TRAIN
# =========================================
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for images, labels, lengths in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(
            images,
            labels,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )

        outputs = outputs.reshape(
            -1,
            NUM_CLASSES
        )

        labels = labels.reshape(-1)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5
        )

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {total_loss:.4f}"
    )

# =========================================
# SAVE
# =========================================
torch.save(
    model.state_dict(),
    os.path.join(
        BASE_DIR,
        "crnn_attention.pth"
    )
)

print("MODEL SAVED: crnn_attention.pth")