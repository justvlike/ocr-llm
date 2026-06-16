import os
import tarfile
import random
import struct
from pathlib import Path

import numpy as np

from PIL import (
    Image,
    ImageFilter
)

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IAM_DIR = os.path.join(
    BASE_DIR,
    "data",
    "IAM",
    "raw"
)

EMNIST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "EMNIST",
    "raw"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRAIN SETTINGS
# =========================
BATCH_SIZE = 2
EPOCHS = 2
LR = 3e-5

MAX_LENGTH = 128

MAX_IAM_SAMPLES = 5000
MAX_EMNIST_SAMPLES = 5000

NUM_WORKERS = 0
PIN_MEMORY = False

# =========================
# AUGMENT MODES
# =========================
# 0 baseline
# 1 gaussian noise
# 2 blur
# 3 rotation
# 4 brightness/contrast
AUGMENT_MODE = 0

# =========================
# CONTINUE TRAINING
# =========================
CONTINUE_FROM_CHECKPOINT = False

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "trocr_iam_emnist.pth"
)

# =========================
# EMNIST MAP
# =========================
EMNIST_CHARS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

# =========================
# EXTRACT IAM
# =========================
def extract_iam():

    lines_dir = Path(IAM_DIR) / "lines"
    ascii_dir = Path(IAM_DIR) / "ascii"

    if not lines_dir.exists():

        print("Extracting lines.tgz ...")

        with tarfile.open(
            Path(IAM_DIR) / "lines.tgz",
            "r:gz"
        ) as tar:

            tar.extractall(IAM_DIR)

    if not ascii_dir.exists():

        print("Extracting ascii.tgz ...")

        with tarfile.open(
            Path(IAM_DIR) / "ascii.tgz",
            "r:gz"
        ) as tar:

            tar.extractall(IAM_DIR)

# =========================
# IAM LOADER
# =========================
def load_iam():

    ascii_file = Path(IAM_DIR) / "ascii" / "lines.txt"

    lines_dir = Path(IAM_DIR) / "lines"

    samples = []

    with open(
        ascii_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 9:
                continue

            img_id = parts[0]
            status = parts[1]

            if status != "ok":
                continue

            text = " ".join(parts[8:])

            folder = img_id.split("-")[0]

            subfolder = (
                f"{img_id.split('-')[0]}-"
                f"{img_id.split('-')[1]}"
            )

            img_path = (
                lines_dir /
                folder /
                subfolder /
                f"{img_id}.png"
            )

            if img_path.exists():

                samples.append(
                    (str(img_path), text)
                )

    return samples[:MAX_IAM_SAMPLES]

# =========================
# READ IDX
# =========================
def read_idx_images(path):

    with open(path, "rb") as f:

        magic, num, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        data = data.reshape(
            num,
            rows,
            cols
        )

    return data

def read_idx_labels(path):

    with open(path, "rb") as f:

        magic, num = struct.unpack(
            ">II",
            f.read(8)
        )

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    return labels

# =========================
# LOAD EMNIST
# =========================
def load_emnist():

    images_path = os.path.join(
        EMNIST_DIR,
        "emnist-balanced-train-images-idx3-ubyte"
    )

    labels_path = os.path.join(
        EMNIST_DIR,
        "emnist-balanced-train-labels-idx1-ubyte"
    )

    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)

    samples = []

    temp_dir = os.path.join(
        BASE_DIR,
        "temp_emnist"
    )

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    for i in range(
        min(MAX_EMNIST_SAMPLES, len(images))
    ):

        img = images[i]

        img = np.transpose(img)

        img = 255 - img

        pil = Image.fromarray(img)

        char = EMNIST_CHARS[
            labels[i] % len(EMNIST_CHARS)
        ]

        save_path = os.path.join(
            temp_dir,
            f"{i}.png"
        )

        pil.save(save_path)

        samples.append(
            (save_path, char)
        )

    return samples

# =========================
# AUGMENTATION
# =========================
def augment_image(image):

    # =========================
    # NOISE
    # =========================
    if AUGMENT_MODE == 1:

        arr = np.array(image).astype(np.float32)

        noise = np.random.normal(
            0,
            10,
            arr.shape
        )

        arr += noise

        arr = np.clip(arr, 0, 255)

        image = Image.fromarray(
            arr.astype(np.uint8)
        )

    # =========================
    # BLUR
    # =========================
    elif AUGMENT_MODE == 2:

        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=0.7
            )
        )

    # =========================
    # ROTATION
    # =========================
    elif AUGMENT_MODE == 3:

        angle = random.uniform(-3, 3)

        image = image.rotate(
            angle,
            fillcolor=(255, 255, 255)
        )

    # =========================
    # BRIGHTNESS / CONTRAST
    # =========================
    elif AUGMENT_MODE == 4:

        arr = np.array(image).astype(np.float32)

        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10, 10)

        arr = alpha * arr + beta

        arr = np.clip(arr, 0, 255)

        image = Image.fromarray(
            arr.astype(np.uint8)
        )

    return image

# =========================
# DATASET
# =========================
class OCRDataset(Dataset):

    def __init__(
        self,
        samples,
        processor
    ):

        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, text = self.samples[idx]

        image = Image.open(
            img_path
        ).convert("RGB")

        image = augment_image(image)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[
            labels ==
            self.processor.tokenizer.pad_token_id
        ] = -100

        return pixel_values, labels

# =========================
# MAIN
# =========================
def main():

    print("\n=========================")
    print("TrOCR Fine-Tuning")
    print("=========================")

    print(f"DEVICE: {DEVICE}")
    print(f"AUGMENT MODE: {AUGMENT_MODE}")

    extract_iam()

    # =========================
    # LOAD DATA
    # =========================
    print("\nLoading IAM...")
    iam_samples = load_iam()

    print("Loading EMNIST...")
    emnist_samples = load_emnist()

    samples = iam_samples + emnist_samples

    random.shuffle(samples)

    split = int(0.9 * len(samples))

    train_samples = samples[:split]
    val_samples = samples[split:]

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # =========================
    # MODEL
    # =========================
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model.config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        processor.tokenizer.eos_token_id
    )

    model.config.decoder_start_token_id = (
        processor.tokenizer.bos_token_id
    )

    # =========================
    # LOAD CHECKPOINT
    # =========================
    if (
        CONTINUE_FROM_CHECKPOINT and
        os.path.exists(CHECKPOINT_PATH)
    ):

        print("\nLoading checkpoint...")

        model.load_state_dict(
            torch.load(
                CHECKPOINT_PATH,
                map_location=DEVICE
            )
        )

        print("Checkpoint loaded")

    model.to(DEVICE)

    # =========================
    # DATASET
    # =========================
    train_dataset = OCRDataset(
        train_samples,
        processor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # =========================
    # OPTIMIZER
    # =========================
    optimizer = AdamW(
        model.parameters(),
        lr=LR
    )

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        )

        for pixel_values, labels in loop:

            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(
                loss=float(loss.item())
            )

        print(
            f"\nEpoch {epoch+1} "
            f"Loss: {total_loss:.4f}"
        )

    # =========================
    # SAVE
    # =========================
    save_name = (
        f"trocr_aug_mode_{AUGMENT_MODE}.pth"
    )

    save_path = os.path.join(
        BASE_DIR,
        save_name
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print("\n=========================")
    print("MODEL SAVED")
    print(save_path)
    print("=========================")

if __name__ == "__main__":
    main()