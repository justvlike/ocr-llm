from pathlib import Path
import json
import random

import pandas as pd
import torch

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "dess" / "dataset" / "test_trocr" / "dataset.csv"
CHECKPOINT_DIR = ROOT / "checkpoints" / "trocr_finetuned"

OUT_DIR = ROOT / "results" / "trocr_finetuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

# можно убрать sample если нужен полный тест
test_df = df.sample(n=1000, random_state=42)

# ============================================================
# MODEL
# ============================================================

processor = TrOCRProcessor.from_pretrained(CHECKPOINT_DIR)

model = VisionEncoderDecoderModel.from_pretrained(
    CHECKPOINT_DIR
).to(DEVICE)

model.eval()

# ============================================================
# EVALUATION
# ============================================================

gts = []
preds = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

    image_path = Path(row["image"])
    image = Image.open(image_path).convert("RGB")

    gt = str(row["text"]).strip()

    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64
        )

    pred = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    gts.append(gt)
    preds.append(pred)

# ============================================================
# METRICS
# ============================================================

metrics = {
    "samples": len(gts),
    "cer": cer(gts, preds),
    "wer": wer(gts, preds),
    "exact_match": sum(
        gt == pr
        for gt, pr in zip(gts, preds)
    ) / len(gts)
}

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame({
    "gt": gts,
    "pred": preds
})

results_df.to_csv(
    OUT_DIR / "predictions.csv",
    index=False
)

with open(
    OUT_DIR / "metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

# ============================================================
# PRINT REPORT
# ============================================================

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(metrics)

print()
print("SAMPLE PREDICTIONS")
print()

# сначала покажем ошибки
errors = [
    (gt, pr)
    for gt, pr in zip(gts, preds)
    if gt != pr
]

random.shuffle(errors)

for gt, pr in errors[:5]:

    print(f"GT: {gt}")
    print(f"PR: {pr}")
    print("-" * 40)

# если ошибок мало — добиваем правильными
if len(errors) < 5:

    correct = [
        (gt, pr)
        for gt, pr in zip(gts, preds)
        if gt == pr
    ]

    for gt, pr in correct[:5 - len(errors)]:

        print(f"GT: {gt}")
        print(f"PR: {pr}")
        print("-" * 40)