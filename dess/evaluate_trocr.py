from pathlib import Path
import json
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dess" / "dataset" / "test" / "dataset.csv"
OUT_DIR = ROOT / "results" / "trocr_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CSV_PATH)
test_df = df.sample(n=25, random_state=42)
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
).to(DEVICE)
model.eval()

gts, preds = [], []

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

pd.DataFrame({
    "gt": gts,
    "pred": preds
}).to_csv(OUT_DIR / "predictions.csv", index=False)
metrics = {
    "samples": len(gts),
    "cer": cer(gts, preds),
    "wer": wer(gts, preds),
    "exact_match": sum(g == p for g, p in zip(gts, preds)) / len(gts)
}

with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(json.dumps(metrics, indent=4))