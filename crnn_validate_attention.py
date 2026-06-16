import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ocr_utils import (
    OCRDataset,
    DEVICE,
    cer,
    wer,
    idx_to_char,
    char_to_idx
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

val_csv = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "val.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "crnn_attention.pth"
)

BATCH_SIZE = 16

MAX_TEXT_LEN = 32

HIDDEN_SIZE = 256
EMBED_SIZE = 128

SOS_TOKEN = len(char_to_idx) + 1
EOS_TOKEN = len(char_to_idx) + 2

NUM_CLASSES = len(char_to_idx) + 3

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

    lengths = torch.tensor(lengths)

    return images, padded_labels, lengths

# =========================================
# DATA
# =========================================
val_dataset = OCRDataset(
    val_csv,
    base_dir=BASE_DIR,
    max_len=MAX_TEXT_LEN
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

        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
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
            embed_size + hidden_size,
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
            input_size=256 * 8,
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
            num_classes,
            HIDDEN_SIZE,
            EMBED_SIZE
        )

    def forward(self, images):
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

        input_char = torch.full(
            (batch_size,),
            SOS_TOKEN,
            dtype=torch.long
        ).to(DEVICE)

        predictions = []

        for _ in range(MAX_TEXT_LEN):
            output, hidden, cell = self.decoder(
                input_char,
                hidden,
                cell,
                encoder_outputs
            )

            best_guess = output.argmax(1)

            predictions.append(best_guess)

            input_char = best_guess

        predictions = torch.stack(
            predictions,
            dim=1
        )

        return predictions

# =========================================
# INIT MODEL
# =========================================
model = CRNNAttention(NUM_CLASSES).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# =========================================
# DECODE
# =========================================
def decode_attention(predictions):
    texts = []

    for seq in predictions:
        chars = []

        for token in seq:
            token = token.item()

            if token == EOS_TOKEN:
                break

            if token in [SOS_TOKEN]:
                continue

            chars.append(
                idx_to_char.get(token, "")
            )

        texts.append("".join(chars))

    return texts

# =========================================
# VALIDATION
# =========================================
all_cer = []
all_wer = []

with torch.no_grad():
    for images, labels, lengths in val_loader:
        images = images.to(DEVICE)

        predictions = model(images)

        pred_texts = decode_attention(
            predictions
        )

        gt_texts = []

        for seq in labels:
            chars = []

            for token in seq:
                token = token.item()

                if token in [SOS_TOKEN, EOS_TOKEN]:
                    continue

                chars.append(
                    idx_to_char.get(token, "")
                )

            gt_texts.append("".join(chars))

        for pred, gt in zip(pred_texts, gt_texts):
            all_cer.append(cer(pred, gt))
            all_wer.append(wer(pred, gt))

print(f"Validation CER: {sum(all_cer)/len(all_cer):.4f}")
print(f"Validation WER: {sum(all_wer)/len(all_wer):.4f}")

# =========================================
# VISUALIZATION
# =========================================
sample = [val_dataset[i] for i in range(10, 15)]

plt.figure(figsize=(20, 4))

for i, (img, label, length) in enumerate(sample):
    ax = plt.subplot(1, 5, i + 1)

    ax.imshow(
        img.squeeze().numpy(),
        cmap="gray"
    )

    ax.axis("off")

    with torch.no_grad():
        pred = model(
            img.unsqueeze(0).to(DEVICE)
        )

        pred_text = decode_attention(pred)[0]

    true_text = "".join([
        idx_to_char.get(l.item(), "")
        for l in label
    ])

    ax.set_title(
        f"P: {pred_text[:20]}\nT: {true_text[:20]}",
        fontsize=8
    )

plt.tight_layout()
plt.show()

print("Validation done")