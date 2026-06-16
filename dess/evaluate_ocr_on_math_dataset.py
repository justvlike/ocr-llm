import re
from pathlib import Path

import pandas as pd
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from jiwer import wer
import editdistance

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from model_crnn import CRNN


# =====================================================
# CONFIG
# =====================================================

OCR_MODEL = "crnn"
# trocr_base
# trocr_finetuned
# crnn

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "dess" / "dataset" / "dataset.csv"

TROCR_FINETUNED_PATH = (
    ROOT / "checkpoints" / "trocr_finetuned"
)

CRNN_PATH = (
    ROOT / "weights" / "crnn_best.pt"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================
# HELPERS
# =====================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def cer(gt, pred):

    gt = normalize_text(gt)
    pred = normalize_text(pred)

    if len(gt) == 0:
        return 0

    return editdistance.eval(gt, pred) / len(gt)


def resize_keep_ratio(
    image,
    width=128,
    height=32,
):

    w, h = image.size

    scale = height / h

    new_w = int(w * scale)

    image = image.resize(
        (new_w, height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "L",
        (width, height),
        255,
    )

    if new_w <= width:
        canvas.paste(image, (0, 0))
    else:
        canvas = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    return np.array(canvas)


# =====================================================
# TROCR
# =====================================================

def load_trocr_base():

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


def load_trocr_finetuned():

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        str(TROCR_FINETUNED_PATH)
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


def trocr_predict(
    processor,
    model,
    image_path,
):

    image = Image.open(
        image_path
    ).convert("RGB")

    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    generated_ids = model.generate(
        pixel_values
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return text


# =====================================================
# CRNN
# =====================================================

def load_crnn():

    checkpoint = torch.load(
        str(CRNN_PATH),
        map_location=DEVICE,
    )

    charset = checkpoint["charset"]

    model = CRNN(
        img_h=32,
        n_channels=1,
        n_classes=len(charset) + 1,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(DEVICE)
    model.eval()

    return model, charset


def ctc_decode(preds, charset):

    preds = preds.argmax(2)

    text = ""

    prev = -1

    for p in preds:

        p = p.item()

        if p != prev and p != 0:
            text += charset[p - 1]

        prev = p

    return text


def crnn_predict(
    model,
    charset,
    image_path,
):

    image = Image.open(
        image_path
    ).convert("L")

    image = resize_keep_ratio(
        image,
        width=128,
        height=32,
    )

    tensor = (
        torch.tensor(
            image,
            dtype=torch.float32
        )
        / 255.0
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        preds = model(tensor)

    preds = preds.squeeze(1)

    return ctc_decode(
        preds,
        charset,
    )


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading OCR...")

if OCR_MODEL == "trocr_base":

    processor, model = load_trocr_base()

elif OCR_MODEL == "trocr_finetuned":

    processor, model = load_trocr_finetuned()

elif OCR_MODEL == "crnn":

    model, charset = load_crnn()

else:
    raise ValueError("Unknown OCR_MODEL")


# =====================================================
# DATA
# =====================================================

df = pd.read_csv(CSV_PATH)
df = df.head(100)

total = len(df)

cer_sum = 0
wer_sum = 0
exact = 0

samples = []


# =====================================================
# EVAL
# =====================================================

for _, row in tqdm(
    df.iterrows(),
    total=total,
):

    image_path = row["image"]

    gt = normalize_text(
        row["text"]
    )

    if OCR_MODEL.startswith("trocr"):

        pred = trocr_predict(
            processor,
            model,
            image_path,
        )

    else:

        pred = crnn_predict(
            model,
            charset,
            image_path,
        )

    pred = normalize_text(pred)

    cer_sum += cer(gt, pred)
    wer_sum += wer(gt, pred)

    if gt == pred:
        exact += 1

    if len(samples) < 20 and gt != pred:

        samples.append(
            (
                gt,
                pred,
            )
        )


# =====================================================
# RESULTS
# =====================================================

results = {
    "ocr_model": OCR_MODEL,
    "samples": total,
    "cer": cer_sum / total,
    "wer": wer_sum / total,
    "exact_match": exact / total,
}

print("\n====================")
print("OCR RESULTS")
print("====================")
print(results)

print("\nSAMPLE ERRORS\n")

for gt, pred in samples:

    print("GT :", gt)
    print("PR :", pred)
    print("-" * 60)