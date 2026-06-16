import sys
from pathlib import Path

import torch
from tqdm import tqdm
import editdistance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(r"E:\Files\PycharmProjects\ocr-llm")

sys.path.append(str(ROOT / "parseq"))

from strhub.models.utils import load_from_checkpoint
from strhub.data.module import SceneTextDataModule

CKPT = ROOT / r"parseq\outputs\parseq\2026-06-12_19-41-39\checkpoints\last.ckpt"

print("Loading model...")

model = load_from_checkpoint(str(CKPT))
model = model.eval().to(DEVICE)

print("Model loaded.")

dm = SceneTextDataModule(
    root_dir=str(ROOT / "data"),
    train_dir="",
    img_size=(32, 128),
    max_label_length=25,
    charset_train="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    charset_test="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    batch_size=32,
    num_workers=0,
    augment=False,
)

val_ds = dm.val_dataset

print("VAL:", len(val_ds))

total_cer = 0
total_words = 0

for i in tqdm(range(len(val_ds))):

    image, gt = val_ds[i]

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = model(image).softmax(-1)
        pred, _ = model.tokenizer.decode(probs)

    pred = pred[0]

    if i < 20:
        print()
        print("GT :", gt)
        print("PR :", pred)

    cer = editdistance.eval(pred, gt) / max(len(gt), 1)

    total_cer += cer
    total_words += 1

print()
print("==========")
print("CER =", total_cer / total_words)
print("SAMPLES =", total_words)