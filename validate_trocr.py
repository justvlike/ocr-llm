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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
MAX_LENGTH = 64

MAX_IAM_SAMPLES = 2000
MAX_EMNIST_SAMPLES = 2000

# =========================
# EMNIST CHARS
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
# LOAD IAM
# =========================
def load_iam():

    ascii_file = Path(IAM_DIR) / "ascii" / "lines.txt"
    lines_dir = Path(IAM_DIR) / "lines"

    samples = []

    with open(ascii_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
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

                samples.append((
                    str(img_path),
                    text
                ))

    samples = samples[:MAX_IAM_SAMPLES]

    return samples

# =========================
# IDX READERS
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
        "temp_emnist_validate"
    )

    os.makedirs(temp_dir, exist_ok=True)

    for i in range(
        min(MAX_EMNIST_SAMPLES, len(images))
    ):

        img = images[i]

        # IMPORTANT:
        # EMNIST is rotated
        img = np.transpose(img)

        # invert colors
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

        samples.append((
            save_path,
            char
        ))

    return samples

# =========================
# DATASET
# =========================
class OCRDataset(Dataset):

    def __init__(self, samples, processor):

        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, text = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

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

    extract_iam()

    # =========================
    # LOAD DATA
    # =========================
    print("Loading IAM...")
    iam_samples = load_iam()

    print("Loading EMNIST...")
    emnist_samples = load_emnist()

    samples = iam_samples + emnist_samples

    random.shuffle(samples)

    print(f"Total samples: {len(samples)}")

    # =========================
    # MODEL
    # =========================
    print("Loading TrOCR...")

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded")

    # =========================
    # DATASET
    # =========================
    dataset = OCRDataset(
        samples,
        processor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE
    )

    # =========================
    # VALIDATION
    # =========================
    all_cer = []
    all_wer = []

    predictions = []
    ground_truths = []
    image_paths = []

    print("Running validation...")

    with torch.no_grad():

        for pixel_values, texts, paths in tqdm(loader):

            pixel_values = pixel_values.to(DEVICE)

            generated_ids = model.generate(
                pixel_values,
                max_length=MAX_LENGTH
            )

            preds = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            for pred, true, path in zip(
                preds,
                texts,
                paths
            ):

                pred = pred.strip()
                true = true.strip()

                predictions.append(pred)
                ground_truths.append(true)
                image_paths.append(path)

                all_cer.append(
                    cer(pred, true)
                )

                all_wer.append(
                    wer(pred, true)
                )

    # =========================
    # METRICS
    # =========================
    mean_cer = sum(all_cer) / len(all_cer)
    mean_wer = sum(all_wer) / len(all_wer)

    print("\n======================")
    print(f"Validation CER: {mean_cer:.4f}")
    print(f"Validation WER: {mean_wer:.4f}")
    print("======================\n")

    # =========================
    # VISUALIZATION
    # =========================
    print("VISUALIZATION")

    vis_ids = random.sample(
        range(len(samples)),
        5
    )

    plt.figure(figsize=(20, 5))

    for i, idx in enumerate(vis_ids):

        image = Image.open(
            image_paths[idx]
        ).convert("RGB")

        ax = plt.subplot(1, 5, i + 1)

        ax.imshow(image)

        ax.axis("off")

        pred = predictions[idx]
        true = ground_truths[idx]

        pred_disp = pred[:25]
        true_disp = true[:25]

        ax.set_title(
            f"P: {pred_disp}\nT: {true_disp}",
            fontsize=8
        )

        print("-" * 50)
        print(f"T: {true}")
        print(f"P: {pred}")

    plt.tight_layout()
    plt.show()

    print("\nVALIDATION DONE")

if __name__ == "__main__":
    main()