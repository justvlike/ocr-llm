import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ocr_utils import OCRDataset, DEVICE, greedy_decode, cer, wer, idx_to_char
from train_improved import CRNNImproved

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crnn_improved.pth")
BATCH_SIZE = 32

# Custom collate
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)          # (B, 1, H, W)
    labels = torch.cat(labels)            # (sum(len(labels)))
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths

# Dataset
val_dataset = OCRDataset(val_csv, base_dir=BASE_DIR)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# Model init
model = CRNNImproved(num_classes=len(idx_to_char) + 1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Validation
all_cer = []
all_wer = []

with torch.no_grad():
    for images, labels, label_lengths in val_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        preds = greedy_decode(outputs)

        split_labels = []
        idx = 0
        for length in label_lengths:
            text = "".join([idx_to_char.get(l.item(), "") for l in labels[idx:idx+length]])
            split_labels.append(text)
            idx += length

        for p, t in zip(preds, split_labels):
            all_cer.append(cer(p, t))
            all_wer.append(wer(p, t))

print(f"Validation CER: {sum(all_cer)/len(all_cer):.4f}")
print(f"Validation WER: {sum(all_wer)/len(all_wer):.4f}")

# Visualization
sample = [val_dataset[i] for i in range(6, 11)]

plt.figure(figsize=(20, 4))

for i, (img, label, length) in enumerate(sample):
    img_np = img.squeeze().numpy()

    ax = plt.subplot(1, 5, i + 1)
    ax.imshow(img_np, cmap="gray")
    ax.axis("off")

    with torch.no_grad():
        pred_text = greedy_decode(model(img.unsqueeze(0).to(DEVICE)))[0]

    true_text = "".join([idx_to_char.get(l.item(), "") for l in label])

    max_len = 25
    pred_text_disp = pred_text[:max_len] + ("…" if len(pred_text) > max_len else "")
    true_text_disp = true_text[:max_len] + ("…" if len(true_text) > max_len else "")

    ax.set_title(f"P: {pred_text_disp}\nT: {true_text_disp}", fontsize=8)

plt.tight_layout()
plt.show()

print("Validation done")