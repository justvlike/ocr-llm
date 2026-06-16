import pandas as pd
import torch
from tqdm import tqdm

# ==================================================
# CONFIG
# ==================================================

MODEL_TYPE = "flan"  # "flan" or "bart"

CSV_PATH = "dataset/dataset.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# LOAD MODEL
# ==================================================

if MODEL_TYPE == "flan":

    from transformers import (
        T5Tokenizer,
        T5ForConditionalGeneration,
    )

    MODEL_NAME = "google/flan-t5-small"
    MODEL_PATH = "flan_nlp.pt"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

elif MODEL_TYPE == "bart":

    from transformers import (
        BartTokenizer,
        BartForConditionalGeneration,
    )

    MODEL_NAME = "facebook/bart-base"
    MODEL_PATH = "bart_nlp.pt"

    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

else:
    raise ValueError("MODEL_TYPE must be 'flan' or 'bart'")

# ==================================================
# LOAD WEIGHTS
# ==================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded")
print("Model:", MODEL_TYPE)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(CSV_PATH)
df = df.head(100)

# для быстрого теста можно ограничить
# df = df.head(100)

total = len(df)

correct_expr = 0
correct_fca = 0

# ==================================================
# EVALUATION
# ==================================================

with torch.no_grad():

    for _, row in tqdm(df.iterrows(), total=total):

        text = str(row["text"]).strip()

        target_expr = str(row["target"]).strip()
        target_result = str(row["result"]).strip()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        output_ids = model.generate(
            **inputs,
            max_length=20,
            num_beams=1,
        )

        pred_expr = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        try:
            pred_result = str(eval(pred_expr))
        except:
            pred_result = "ERR"

        if pred_expr == target_expr:
            correct_expr += 1

        if pred_result == target_result:
            correct_fca += 1

# ==================================================
# RESULTS
# ==================================================

expression_accuracy = correct_expr / total
fca = correct_fca / total

print("\n========================")
print("NLP RESULTS")
print("========================")

print({
    "model": MODEL_TYPE,
    "samples": total,
    "expression_accuracy": expression_accuracy,
    "FCA": fca,
})

print("\nDone.")