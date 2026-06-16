import os
import tarfile
import random
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from tqdm import tqdm

from ocr_utils import cer, wer

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

# =========================
# SELECT MODEL
# =========================
AUGMENT_MODE = 4

MODEL_PATH = os.path.join(
    BASE_DIR,
    f"trocr_aug_mode_{AUGMENT_MODE}.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# VALIDATION SETTINGS
# =========================
BATCH_SIZE = 4
MAX_LENGTH = 128

MAX_IAM_SAMPLES = 500
MAX_EMNIST_SAMPLES = 500

VIS_COUNT = 5

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
# IAM VALIDATION
# =========================
def load_iam_validation():

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

    samples = samples[:5000]

    split = int(0.9 * len(samples))

    val_samples = samples[split:]

    return val_samples[:MAX_IAM_SAMPLES]

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
# LOAD EMNIST VALIDATION
# =========================
def load_emnist_validation():

    images_path = os.path.join(
        EMNIST_DIR,
        "emnist-balanced-test-images-idx3-ubyte"
    )

    labels_path = os.path.join(
        EMNIST_DIR,
        "emnist-balanced-test-labels-idx1-ubyte"
    )

    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)

    samples = []

    temp_dir = os.path.join(
        BASE_DIR,
        "temp_emnist_val"
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
# DATASET
# =========================
class OCRValidationDataset(Dataset):

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

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        return (
            pixel_values,
            text,
            img_path
        )

# =========================
# MAIN
# =========================
def main():

    print("\n=========================")
    print("TrOCR VALIDATION")
    print("=========================")

    print(f"DEVICE: {DEVICE}")
    print(f"AUGMENT MODE: {AUGMENT_MODE}")

    extract_iam()

    # =========================
    # LOAD VALIDATION
    # =========================
    print("\nLoading IAM validation...")
    iam_samples = load_iam_validation()

    print("Loading EMNIST validation...")
    emnist_samples = load_emnist_validation()

    samples = iam_samples + emnist_samples

    random.shuffle(samples)

    print(f"\nValidation samples: {len(samples)}")

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

    print("\nLoading fine-tuned weights...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    # =========================
    # DATASET
    # =========================
    dataset = OCRValidationDataset(
        samples,
        processor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =========================
    # VALIDATION
    # =========================
    all_cer = []
    all_wer = []

    print("\nRunning validation...")

    with torch.no_grad():

        for pixel_values, texts, _ in tqdm(loader):

            pixel_values = pixel_values.to(DEVICE)

            generated_ids = model.generate(
                pixel_values,
                max_length=MAX_LENGTH
            )

            preds = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            for pred, true in zip(preds, texts):

                pred = pred.strip()
                true = true.strip()

                all_cer.append(
                    cer(pred, true)
                )

                all_wer.append(
                    wer(pred, true)
                )

    # =========================
    # METRICS
    # =========================
    mean_cer = (
        sum(all_cer) / len(all_cer)
    )

    mean_wer = (
        sum(all_wer) / len(all_wer)
    )

    print("\n======================")
    print(f"Validation CER: {mean_cer:.4f}")
    print(f"Validation WER: {mean_wer:.4f}")
    print("======================")

    # =========================
    # VISUALIZATION
    # =========================
    print("\nVISUALIZATION")

    sample_vis = random.sample(
        samples,
        VIS_COUNT
    )

    plt.figure(figsize=(20, 5))

    for i, (img_path, true_text) in enumerate(sample_vis):

        image = Image.open(
            img_path
        ).convert("RGB")

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(DEVICE)

        generated = model.generate(
            pixel_values,
            max_length=MAX_LENGTH
        )

        pred = processor.decode(
            generated[0],
            skip_special_tokens=True
        )

        ax = plt.subplot(
            1,
            VIS_COUNT,
            i + 1
        )

        ax.imshow(image)

        ax.axis("off")

        pred_disp = pred[:25]
        true_disp = true_text[:25]

        ax.set_title(
            f"P: {pred_disp}\nT: {true_disp}",
            fontsize=8
        )

        print("-" * 50)
        print(f"T: {true_text}")
        print(f"P: {pred}")

    plt.tight_layout()
    plt.show()

    print("\nVALIDATION DONE")

if __name__ == "__main__":
    main()