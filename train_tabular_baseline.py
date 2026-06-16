import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from ocr_utils import (
    OCRDataset,
    CRNN,
    DEVICE,
    greedy_decode,
    cer,
    wer,
    idx_to_char,
    char_to_idx
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")

MODEL_PATH = os.path.join(BASE_DIR, "crnn_baseline.pth")

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "ocr_tabular_dataset.csv"
)

# =========================
# SETTINGS
# =========================
BATCH_SIZE = 16

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
# FEATURE EXTRACTOR
# =========================
def extract_features(pred_text, probs):
    text_len = len(pred_text)

    digit_count = sum(c.isdigit() for c in pred_text)
    alpha_count = sum(c.isalpha() for c in pred_text)
    space_count = sum(c.isspace() for c in pred_text)

    unique_chars = len(set(pred_text)) if text_len > 0 else 0

    digit_ratio = digit_count / text_len if text_len > 0 else 0
    alpha_ratio = alpha_count / text_len if text_len > 0 else 0
    space_ratio = space_count / text_len if text_len > 0 else 0

    repeated_chars = sum(
        pred_text[i] == pred_text[i - 1]
        for i in range(1, text_len)
    )

    repeat_ratio = repeated_chars / text_len if text_len > 0 else 0

    avg_conf = float(np.mean(probs)) if len(probs) > 0 else 0
    max_conf = float(np.max(probs)) if len(probs) > 0 else 0
    min_conf = float(np.min(probs)) if len(probs) > 0 else 0
    std_conf = float(np.std(probs)) if len(probs) > 0 else 0

    return {
        "text_length": text_len,

        "digit_count": digit_count,
        "alpha_count": alpha_count,
        "space_count": space_count,

        "digit_ratio": digit_ratio,
        "alpha_ratio": alpha_ratio,
        "space_ratio": space_ratio,

        "unique_chars": unique_chars,

        "repeat_ratio": repeat_ratio,

        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
        "min_confidence": min_conf,
        "std_confidence": std_conf
    }

# =========================
# DATA
# =========================
val_dataset = OCRDataset(
    val_csv,
    base_dir=BASE_DIR
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# =========================
# MODEL
# =========================
model = CRNN(
    num_classes=len(char_to_idx) + 1,
    hidden_size=512
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

# =========================
# GENERATE DATASET
# =========================
rows = []

with torch.no_grad():
    for images, labels, label_lengths in val_loader:
        images = images.to(DEVICE)

        outputs = model(images)

        probs_tensor = outputs.softmax(2)

        preds = greedy_decode(outputs)

        # =========================
        # SPLIT LABELS
        # =========================
        split_labels = []

        idx = 0

        for length in label_lengths:
            true_text = "".join([
                idx_to_char.get(l.item(), "")
                for l in labels[idx:idx + length]
            ])

            split_labels.append(true_text)

            idx += length

        # =========================
        # FEATURES
        # =========================
        for batch_idx, (pred_text, true_text) in enumerate(
            zip(preds, split_labels)
        ):
            timestep_probs = probs_tensor[:, batch_idx, :]

            max_probs = torch.max(
                timestep_probs,
                dim=1
            ).values.cpu().numpy()

            features = extract_features(
                pred_text,
                max_probs
            )

            features["pred_text"] = pred_text
            features["true_text"] = true_text

            features["cer"] = cer(pred_text, true_text)
            features["wer"] = wer(pred_text, true_text)

            features["is_exact"] = int(
                pred_text == true_text
            )

            rows.append(features)

# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(df.head())

print()
print(f"DATASET SAVED: {OUTPUT_CSV}")
print(f"ROWS: {len(df)}")