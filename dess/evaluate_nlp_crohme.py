import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# =========================
MODEL_PATH = "flan_nlp.pt"
MODEL_NAME = "google/flan-t5-small"
CSV_PATH = "data/crohme.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded")


# =========================
df = pd.read_csv(CSV_PATH)

correct = 0
total = len(df)


# =========================
with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=total):

        # CROHME is already expression (LaTeX style)
        text = row["expression"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(DEVICE)

        output = model.generate(
            **inputs,
            max_length=64,
            num_beams=1
        )

        pred = tokenizer.decode(output[0], skip_special_tokens=True)

        if pred.strip() == str(text).strip():
            correct += 1


# =========================
print("\nCROHME RESULTS")
print({
    "samples": total,
    "expression_accuracy": correct / total
})