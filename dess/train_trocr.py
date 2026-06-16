from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

import evaluate

# =========================
# PATHS
# =========================

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "dess" / "dataset" / "train" / "dataset.csv"
CHECKPOINT_DIR = ROOT / "checkpoints" / "trocr_finetuned"

# =========================
# MODEL
# =========================

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# =========================
# DATASET
# =========================

class MathDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image"])

        image = Image.open(image_path).convert("RGB")

        pixel_values = processor(
            image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # target expression (what we actually want OCR to output)
        text = str(row["text"]).strip()

        labels = processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
        ).input_ids

        labels = [
            t if t != processor.tokenizer.pad_token_id else -100
            for t in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# =========================
# LOAD DATA
# =========================

train_dataset = MathDataset(CSV_PATH)
val_dataset = MathDataset(CSV_PATH)  # можно потом split добавить

# =========================
# METRIC
# =========================

cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids = np.where(
        label_ids == -100,
        processor.tokenizer.pad_token_id,
        label_ids,
    )

    pred_str = processor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )

    label_str = processor.batch_decode(
        label_ids,
        skip_special_tokens=True
    )

    cer_value = cer_metric.compute(
        predictions=pred_str,
        references=label_str,
    )

    return {"cer": cer_value}

# =========================
# TRAIN ARGS
# =========================

training_args = Seq2SeqTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

    predict_with_generate=True,

    eval_strategy="no",   # быстрее (можешь включить позже)
    save_strategy="epoch",
    logging_strategy="epoch",

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=2,

    learning_rate=3e-5,
    weight_decay=0.01,

    fp16=torch.cuda.is_available(),

    save_total_limit=2,
    report_to="none",
)

# =========================
# TRAINER
# =========================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(CHECKPOINT_DIR)
processor.save_pretrained(CHECKPOINT_DIR)

print("DONE ->", CHECKPOINT_DIR)