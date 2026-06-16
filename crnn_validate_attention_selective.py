import os
import random

import cv2
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Dataset
)

from ocr_utils import (
    OCRDataset,
    DEVICE,
    char_to_idx,
    idx_to_char,
    cer,
    wer
)

# =========================================
# CONFIG
# =========================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

train_csv = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train.csv"
)

val_csv = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "val.csv"
)

IMG_HEIGHT = 32
IMG_WIDTH = 256

BATCH_SIZE = 16

MAX_TEXT_LEN = 16

HIDDEN_SIZE = 256
EMBED_SIZE = 128

SOS_TOKEN = len(char_to_idx) + 1
EOS_TOKEN = len(char_to_idx) + 2

NUM_CLASSES = len(char_to_idx) + 3

# =========================================
# AUGMENT MODES
# =========================================
# 0 = baseline
# 1 = noise
# 2 = blur
# 3 = rotation
# 4 = brightness/contrast
AUGMENT_MODE = 0

MODEL_PATH = os.path.join(
    BASE_DIR,
    f"crnn_attention_aug_{AUGMENT_MODE}.pth"
)

# =========================================
# AUGMENTATION
# =========================================
def augment_image(img_tensor):

    img = img_tensor.squeeze().numpy()

    img = (img * 255).astype(np.uint8)

    # =========================
    # NOISE
    # =========================
    if AUGMENT_MODE == 1:

        noise = np.random.normal(
            0,
            20,
            img.shape
        )

        img = img.astype(np.float32)

        img += noise

        img = np.clip(img, 0, 255)

        img = img.astype(np.uint8)

    # =========================
    # BLUR
    # =========================
    elif AUGMENT_MODE == 2:

        img = cv2.GaussianBlur(
            img,
            (3, 3),
            0
        )

    # =========================
    # ROTATION
    # =========================
    elif AUGMENT_MODE == 3:

        angle = random.uniform(-5, 5)

        h, w = img.shape

        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        img = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            borderValue=255
        )

    # =========================
    # BRIGHTNESS / CONTRAST
    # =========================
    elif AUGMENT_MODE == 4:

        alpha = random.uniform(0.7, 1.3)

        beta = random.randint(-30, 30)

        img = cv2.convertScaleAbs(
            img,
            alpha=alpha,
            beta=beta
        )

    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img).unsqueeze(0)

    return img

# =========================================
# DATASET WRAPPER
# =========================================
class AugmentedOCRDataset(Dataset):

    def __init__(
        self,
        csv_path,
        base_dir,
        max_len
    ):

        self.dataset = OCRDataset(
            csv_path,
            base_dir=base_dir,
            max_len=max_len
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, label, length = self.dataset[idx]

        image = augment_image(image)

        return image, label, length

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

        label_len = min(
            len(label),
            MAX_TEXT_LEN - 1
        )

        padded_labels[i, 0] = SOS_TOKEN

        padded_labels[
            i,
            1:label_len + 1
        ] = label[:label_len]

    lengths = torch.tensor(
        lengths,
        dtype=torch.long
    )

    return images, padded_labels, lengths

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

    def forward(
        self,
        hidden,
        encoder_outputs
    ):

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

        attention = self.v(
            energy
        ).squeeze(2)

        return torch.softmax(
            attention,
            dim=1
        )

# =========================================
# DECODER
# =========================================
class AttentionDecoder(nn.Module):

    def __init__(
        self,
        output_size,
        hidden_size,
        embed_size
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            output_size,
            embed_size
        )

        self.attention = Attention(
            hidden_size
        )

        self.rnn = nn.LSTM(
            hidden_size + embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(
        self,
        input_char,
        hidden,
        cell,
        encoder_outputs
    ):

        input_char = input_char.unsqueeze(1)

        embedded = self.embedding(
            input_char
        )

        attention_weights = self.attention(
            hidden[-1],
            encoder_outputs
        )

        attention_weights = (
            attention_weights.unsqueeze(1)
        )

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

            nn.Conv2d(
                1,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                128,
                256,
                3,
                padding=1
            ),

            nn.ReLU()
        )

        self.encoder = nn.LSTM(

            input_size=256 * (IMG_HEIGHT // 4),

            hidden_size=HIDDEN_SIZE,

            num_layers=1,

            bidirectional=True,

            batch_first=True
        )

        self.reduce = nn.Linear( #self.hidden_reduce
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
        images
    ):

        x = self.cnn(images)

        b, c, h, w = x.size()

        x = x.permute(
            0,
            3,
            1,
            2
        )

        x = x.contiguous().view(
            b,
            w,
            c * h
        )

        encoder_outputs, (hidden, cell) = (
            self.encoder(x)
        )

        encoder_outputs = self.reduce( #self.hidden_reduce
            encoder_outputs
        )

        hidden = torch.tanh(

            self.reduce( #self.hidden_reduce

                torch.cat(
                    (hidden[0], hidden[1]),
                    dim=1
                )
            )

        ).unsqueeze(0)

        cell = torch.zeros_like(hidden)

        batch_size = images.size(0)

        outputs = []

        input_char = torch.full(
            (batch_size,),
            SOS_TOKEN,
            dtype=torch.long
        ).to(DEVICE)

        for _ in range(MAX_TEXT_LEN):

            output, hidden, cell = self.decoder(
                input_char,
                hidden,
                cell,
                encoder_outputs
            )

            best_guess = output.argmax(1)

            outputs.append(best_guess)

            input_char = best_guess

        outputs = torch.stack(
            outputs,
            dim=1
        )

        return outputs

# =========================================
# LOAD MODEL
# =========================================
model = CRNNAttention(
    NUM_CLASSES
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print(f"Loaded: {MODEL_PATH}")

# =========================================
# DATA
# =========================================
val_dataset = AugmentedOCRDataset(
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
# VALIDATION
# =========================================
all_cer = []
all_wer = []

predictions = []
ground_truths = []

print("Running validation...")

with torch.no_grad():

    for images, labels, lengths in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        for pred_seq, true_seq in zip(outputs, labels):

            pred_text = ""

            for idx in pred_seq:

                idx = idx.item()

                if idx == EOS_TOKEN:
                    break

                if idx in idx_to_char:
                    pred_text += idx_to_char[idx]

            true_text = ""

            for idx in true_seq:

                idx = idx.item()

                if idx == EOS_TOKEN:
                    break

                if idx == SOS_TOKEN:
                    continue

                if idx in idx_to_char:
                    true_text += idx_to_char[idx]

            pred_text = pred_text.lower()
            true_text = true_text.lower()

            predictions.append(pred_text)
            ground_truths.append(true_text)

            all_cer.append(
                cer(pred_text, true_text)
            )

            all_wer.append(
                wer(pred_text, true_text)
            )

# =========================================
# METRICS
# =========================================
mean_cer = (
    sum(all_cer) / len(all_cer)
)

mean_wer = (
    sum(all_wer) / len(all_wer)
)

print()
print(f"Validation CER: {mean_cer:.4f}")
print(f"Validation WER: {mean_wer:.4f}")

# =========================================
# VISUALIZATION
# =========================================
print("\nVISUALIZATION")

plt.figure(figsize=(20, 4))

sample_ids = random.sample(
    range(len(val_dataset)),
    5
)

for i, idx in enumerate(sample_ids):

    img, _, _ = val_dataset[idx]

    img_np = img.squeeze().numpy()

    pred_text = predictions[idx]
    true_text = ground_truths[idx]

    ax = plt.subplot(6, 10, i + 1)

    ax.imshow(
        img_np,
        cmap="gray"
    )

    ax.axis("off")

    ax.set_title(
        f"P: {pred_text[:20]}\n"
        f"T: {true_text[:20]}",
        fontsize=8
    )

    print("-" * 50)
    print(f"T: {true_text}")
    print(f"P: {pred_text}")

plt.tight_layout()
plt.show()

print("\nVALIDATION DONE")