from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import json
import re

from jiwer import cer, wer

from strhub.models.utils import create_model
from strhub.models.utils import get_pretrained_weights
from strhub.data.module import SceneTextDataModule


# -------------------
# CONFIG
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/IAM/processed/iam.csv"

OUT_DIR = ROOT / "results/parseq_final_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------
# DATA
# -------------------
df = pd.read_csv(CSV_PATH)


# -------------------
# MODEL (CORRECT WAY FOR YOUR STRHUB)
# -------------------
print("\nLOADING MODEL...")

model = create_model("parseq", pretrained=False)
model = model.to(DEVICE).eval()


print("LOADING PRETRAINED WEIGHTS EXPLICITLY...")

state = get_pretrained_weights("parseq")
missing, unexpected = model.load_state_dict(state, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))


# -------------------
# TRANSFORM
# -------------------
transform = SceneTextDataModule.get_transform(
    model.hparams.img_size,
    augment=False
)


# -------------------
# NORMALIZE TEXT
# -------------------
def normalize(text: str) -> str:
    text = str(text)
    text = text.replace("|", " ")
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split()).strip()


# -------------------
# RUN
# -------------------
gts, preds = [], []

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):

        image_path = ROOT / str(row["image_path"]).replace("\\", "/")
        image = Image.open(image_path).convert("RGB")

        gt = normalize(row["text"])

        image = transform(image).unsqueeze(0).to(DEVICE)

        logits = model(image)

        decoded = model.tokenizer.decode(logits)

        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0]

        pred = str(decoded).lower().strip()

        gts.append(gt)
        preds.append(pred)


# -------------------
# METRICS
# -------------------
metrics = {
    "samples": len(gts),
    "cer": cer(gts, preds),
    "wer": wer(gts, preds),
    "exact_match": sum(g == p for g, p in zip(gts, preds)) / len(gts)
}


# -------------------
# SAVE
# -------------------
pd.DataFrame({
    "ground_truth": gts,
    "prediction": preds
}).to_csv(OUT_DIR / "predictions.csv", index=False)

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)


# -------------------
# DEBUG
# -------------------
print("\n====================")
print("RESULTS")
print("====================")
print(metrics)

print("\nSAMPLES:\n")

for i in range(10):
    print("GT:", gts[i])
    print("PR:", preds[i])
    print("---")