import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna

from ocr_utils import OCRDataset, CRNN, DEVICE, char_to_idx, greedy_decode, cer, idx_to_char

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_csv = os.path.join(BASE_DIR, "data", "processed", "train.csv")
val_csv = os.path.join(BASE_DIR, "data", "processed", "val.csv")
MAX_TEXT_LEN = 32
EPOCHS = 5

# Custom collate
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths

def objective(trial):

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])

    # Dataset
    train_dataset = OCRDataset(train_csv, base_dir=BASE_DIR, max_len=MAX_TEXT_LEN)
    val_dataset = OCRDataset(val_csv, base_dir=BASE_DIR, max_len=MAX_TEXT_LEN)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    class CRNNTrial(CRNN):
        def __init__(self, num_classes):
            super().__init__(num_classes)
            self.rnn = nn.LSTM(
                input_size=128 * (32 // 4),
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size * 2, num_classes)

    model = CRNNTrial(num_classes=len(char_to_idx) + 1).to(DEVICE)

    criterion = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train loop
    for epoch in range(EPOCHS):
        model.train()

        for images, labels, label_lengths in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            outputs = model(images)

            input_lengths = torch.full(
                (images.size(0),),
                outputs.size(0),
                dtype=torch.long
            ).to(DEVICE)

            loss = criterion(
                outputs.log_softmax(2),
                labels,
                input_lengths,
                label_lengths
            )

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation (CER)
    model.eval()
    all_cer = []

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

    mean_cer = sum(all_cer) / len(all_cer)

    return mean_cer  # minimize

# Study
if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=20)

    print("BEST PARAMS:")
    print(study.best_params)

    print("BEST CER:")
    print(study.best_value)