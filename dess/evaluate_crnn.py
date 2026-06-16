from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from jiwer import cer, wer

from model_crnn import CRNN


# =====================================================
# CONFIG
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT / "dess" / "dataset" / "test" / "dataset.csv"
WEIGHTS = ROOT / "weights" / "crnn_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_H = 32
IMG_W = 512

BATCH_SIZE = 16


# =====================================================
# LOAD CHECKPOINT
# =====================================================

checkpoint = torch.load(
    WEIGHTS,
    map_location=DEVICE,
)

charset = checkpoint["charset"]

idx_to_char = {
    i + 1: c
    for i, c in enumerate(charset)
}

num_classes = len(charset) + 1

model = CRNN(num_classes)

model.load_state_dict(
    checkpoint["model"]
)

model.to(DEVICE)
model.eval()

print("Loaded model")
print("Charset size:", len(charset))


# =====================================================
# DATASET
# =====================================================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])


class MathDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_rel_path = str(row["image"]).replace("\\", "/")

        # вариант 1: путь уже полный относительно ROOT
        image_path = ROOT / image_rel_path

        # вариант 2: в csv лежит dataset/test/...
        if not image_path.exists():
            image_path = ROOT / "dess" / image_rel_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_rel_path}"
            )

        image = Image.open(image_path).convert("L")
        image = transform(image)

        gt = str(row["text"]).strip()

        return image, gt


dataset = MathDataset(TEST_CSV)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =====================================================
# GREEDY CTC DECODE
# =====================================================

def ctc_decode(logits):

    preds = logits.argmax(2)

    results = []

    for seq in preds:

        prev = -1
        chars = []

        for p in seq.cpu().numpy():

            p = int(p)

            if p != prev and p != 0:
                chars.append(
                    idx_to_char.get(p, "")
                )

            prev = p

        results.append(
            "".join(chars)
        )

    return results


# =====================================================
# EVALUATION
# =====================================================

gts = []
preds = []

with torch.no_grad():

    for images, gt_texts in tqdm(loader):

        images = images.to(DEVICE)

        logits = model(images)

        batch_preds = ctc_decode(logits)

        gts.extend(gt_texts)
        preds.extend(batch_preds)


# =====================================================
# METRICS
# =====================================================

cer_score = cer(gts, preds)
wer_score = wer(gts, preds)

exact_match = sum(
    gt == pred
    for gt, pred in zip(gts, preds)
) / len(gts)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print({
    "samples": len(gts),
    "cer": cer_score,
    "wer": wer_score,
    "exact_match": exact_match,
})

print()
print("SAMPLE PREDICTIONS")
print()

for gt, pr in list(zip(gts, preds))[:5]:

    print("GT:", gt)
    print("PR:", pr)
    print("-" * 40)

print(print(set(preds[:100])))

for images, gt in loader:
    print(images.mean(), images.std())
    break

print(len(set(preds)))