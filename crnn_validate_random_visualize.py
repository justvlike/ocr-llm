import os
import random
import matplotlib.pyplot as plt
import torch
from ocr_utils import OCRDataset, CRNN, DEVICE, greedy_decode, idx_to_char

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crnn_baseline.pth")

val_dataset = OCRDataset(val_csv, base_dir=BASE_DIR)

model = CRNN(num_classes=len(idx_to_char)+1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

N = 10
indices = random.sample(range(len(val_dataset)), N)

plt.figure(figsize=(20, 4))

for i, idx in enumerate(indices):
    img, label, length = val_dataset[idx]

    with torch.no_grad():
        pred_text = greedy_decode(model(img.unsqueeze(0).to(DEVICE)))[0]

    true_text = "".join([idx_to_char.get(l.item(), "") for l in label])

    img_np = img.squeeze().cpu().numpy()

    plt.subplot(1, N, i + 1)
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    plt.title(f"P: {pred_text}\nT: {true_text}", fontsize=8)

plt.tight_layout()
plt.show()