import pandas as pd
import torch
from tqdm import tqdm

MODEL_TYPE = "flan"  # "flan" | "bart"
CSV_PATH = "dataset/dataset.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
print("Model loaded")
print("Model:", MODEL_TYPE)

df = pd.read_csv(CSV_PATH)
df = df.head(25)
total = len(df)
correct_expr = 0
correct_fca = 0
valid_calc_samples = 0
zero_div_samples = 0

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=total):
        text = str(row["text"]).strip()
        target_expr = str(row["target"]).strip()
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
        if pred_expr == target_expr:
            correct_expr += 1

        try:
            target_result = float(eval(target_expr))
        except:
            zero_div_samples += 1
            continue
        valid_calc_samples += 1

        try:
            pred_result = float(eval(pred_expr))
        except:
            pred_result = None

        if (
            pred_result is not None
            and abs(pred_result - target_result) < 1e-6
        ):
            correct_fca += 1

        if pred_expr == target_expr:
            if (
                pred_result is None
                or abs(pred_result - target_result) >= 1e-6
            ):
                print("\nBROKEN SAMPLE")
                print("text        :", text)
                print("target_expr :", target_expr)
                print("pred_expr   :", pred_expr)
                print("target_res  :", target_result)
                print("pred_res    :", pred_result)

expression_accuracy = correct_expr / total
if valid_calc_samples > 0:
    fca = correct_fca / valid_calc_samples
else:
    fca = 0
print("\n========================")
print("NLP RESULTS")
print("========================")
import json
print(json.dumps({
    "model": MODEL_TYPE,
    "samples": total,
    "valid_calc_samples": valid_calc_samples,
    "zero_div_samples": zero_div_samples,
    "expression_accuracy": expression_accuracy,
    "FCA": fca,
}, indent=4, ensure_ascii=False))
print("\nDone.")