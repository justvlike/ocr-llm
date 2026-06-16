import difflib
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
)

from model_crnn import CRNN

# =====================================================
# CONFIG
# =====================================================

OCR_MODEL = "trocr"      # trocr | crnn
NLP_MODEL = "flan"       # bart | flan

LOG_ENABLED = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_PATH = "dataset/test_trocr/dataset.csv"

TROCR_PATH = r"E:\Files\PycharmProjects\ocr-llm\checkpoints\trocr_finetuned"
CRNN_PATH = r"E:\Files\PycharmProjects\ocr-llm\weights\crnn_best.pt"

FLAN_PATH = "flan_nlp.pt"
BART_PATH = "bart_nlp.pt"

SAMPLES = 25

OUT_CSV = "correction_results.csv"

WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}

# =====================================================
# VALID VOCAB
# =====================================================

NUMBER_WORDS = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight",
    "nine", "ten"
]

OPERATOR_WORDS = [
    "plus",
    "minus",
    "times",
    "divided",
    "by"
]

VALID_WORDS = NUMBER_WORDS + OPERATOR_WORDS

# =====================================================
# CRNN TRANSFORM
# =====================================================

crnn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 512)),
    transforms.ToTensor(),
])

# =====================================================
# OCR : TROCR
# =====================================================

def load_trocr():

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        TROCR_PATH
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


def trocr_predict(processor, model, image_path):

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            pixel_values,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True
    )[0].strip()

    confs = []

    for score in outputs.scores:

        probs = torch.softmax(score, dim=-1)

        confs.append(
            probs.max().item()
        )

    confidence = (
        sum(confs) / len(confs)
        if len(confs) > 0 else 0
    )

    return text, confidence

# =====================================================
# OCR : CRNN
# =====================================================

def load_crnn():

    checkpoint = torch.load(
        CRNN_PATH,
        map_location=DEVICE,
    )

    charset = checkpoint["charset"]

    model = CRNN(
        len(charset) + 1
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(DEVICE)
    model.eval()

    return model, charset


def ctc_decode(logits, charset):

    pred = logits.argmax(2)[0]

    text = ""

    prev = -1

    for p in pred:

        p = p.item()

        if p != prev and p != 0:
            text += charset[p - 1]

        prev = p

    return text


def crnn_predict(model, charset, image_path):

    image = Image.open(
        image_path
    ).convert("L")

    image = crnn_transform(image)

    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():

        logits = model(image)

        probs = torch.softmax(
            logits,
            dim=2
        )

        confidence = (
            probs.max(dim=2)
            .values
            .mean()
            .item()
        )

    text = ctc_decode(
        logits,
        charset
    )

    return text.strip(), confidence

# =====================================================
# NLP : FLAN
# =====================================================

def load_flan():

    model_name = "google/flan-t5-small"

    tokenizer = T5Tokenizer.from_pretrained(
        model_name
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_name
    )

    model.load_state_dict(
        torch.load(
            FLAN_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model

# =====================================================
# NLP : BART
# =====================================================

def load_bart():

    model_name = "facebook/bart-base"

    tokenizer = BartTokenizer.from_pretrained(
        model_name
    )

    model = BartForConditionalGeneration.from_pretrained(
        model_name
    )

    model.load_state_dict(
        torch.load(
            BART_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model

# =====================================================
# NLP
# =====================================================

def nlp_predict(tokenizer, model, text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_length=20,
        num_beams=1,
    )

    pred = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return pred.strip()

# =====================================================
# ERROR TYPE
# =====================================================

import re

OPS = {"+", "-", "*", "/"}

WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}

def extract_numbers(tokens):
    nums = []
    for t in tokens:
        if t.isdigit():
            nums.append(int(t))
        elif t in WORD_TO_NUM:
            nums.append(WORD_TO_NUM[t])
    return nums

def normalize_expr(expr: str):
    expr = expr.lower().strip()
    expr = expr.replace("by", " by ")
    expr = expr.replace("divided by", "/")
    expr = expr.replace("times", "*")
    expr = expr.replace("plus", "+")
    expr = expr.replace("minus", "-")
    expr = re.sub(r"\s+", " ", expr)
    return expr


def extract_tokens(expr: str):
    expr = normalize_expr(expr)
    return re.findall(r"[a-z]+|\d+|[+\-*/]", expr)


def extract_structure(tokens):
    ops = [t for t in tokens if t in OPS]
    nums = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if t.isalpha()]
    return ops, nums, words


def classify_error(gt: str, pred: str):

    gt_t = extract_tokens(gt)
    pr_t = extract_tokens(pred)

    # -----------------------------
    # 0. GARBAGE
    # -----------------------------
    if len(pr_t) == 0:
        return "GARBAGE"

    # -----------------------------
    # 1. PERFECT
    # -----------------------------
    if gt_t == pr_t:
        return "PERFECT"

    gt_ops = [t for t in gt_t if t in OPS]
    pr_ops = [t for t in pr_t if t in OPS]

    gt_nums = extract_numbers(gt_t)
    pr_nums = extract_numbers(pr_t)

    # -----------------------------
    # 2. OPERATOR / STRUCTURE ERRORS
    # -----------------------------
    if gt_ops != pr_ops:

        if len(gt_ops) == len(pr_ops):
            return "SEVERE_SEMANTIC_ERROR(op_flip)"

        elif len(gt_ops) > len(pr_ops):
            return "SEVERE_SEMANTIC_ERROR(op_missing)"

        else:
            return "SEVERE_SEMANTIC_ERROR(op_extra)"

    # -----------------------------
    # 3. DIVISION BY ZERO (semantic critical)
    # -----------------------------
    if "/" in gt_ops:
        try:
            gt_zero_div = ("0" in gt_nums and len(gt_nums) > 0 and gt_nums[-1] == "0")
            pr_zero_div = ("0" in pr_nums and len(pr_nums) > 0 and pr_nums[-1] == "0")

            if gt_zero_div != pr_zero_div:
                return "SEVERE_SEMANTIC_ERROR(zero_div)"
        except:
            pass

    # -----------------------------
    # 4. NUMERIC ERRORS
    # -----------------------------
    if gt_nums != pr_nums:

        # structural mismatch (count differs)
        if len(gt_nums) != len(pr_nums):
            return "NUMERIC_ERROR(structural)"

        # lexical (small digit corruption)
        digit_diff = sum(g != p for g, p in zip(gt_nums, pr_nums))

        if digit_diff == 1:
            return "NUMERIC_ERROR(lexical)"

        # value mismatch (everything aligns but values wrong)
        return "NUMERIC_ERROR(value)"

    # -----------------------------
    # 5. OCR / TOKEN NOISE
    # -----------------------------
    gt_set = set(gt_t)
    pr_set = set(pr_t)

    if len(gt_set.symmetric_difference(pr_set)) <= 2:
        return "COSMETIC(ocr_noise)"

    return "COSMETIC(unknown)"

# =====================================================
# DICTIONARY CORRECTION
# =====================================================

def dictionary_correct(text):

    corrected = []

    for token in text.split():

        if token in VALID_WORDS:
            corrected.append(token)
            continue

        match = difflib.get_close_matches(
            token,
            VALID_WORDS,
            n=1,
            cutoff=0.6
        )

        if len(match):
            corrected.append(match[0])
        else:
            corrected.append(token)

    return " ".join(corrected)

# =====================================================
# LOAD MODELS
# =====================================================

print("Loading OCR...")

if OCR_MODEL == "trocr":

    ocr_processor, ocr_model = load_trocr()

elif OCR_MODEL == "crnn":

    ocr_model, charset = load_crnn()

else:
    raise ValueError()

print("Loading NLP...")

if NLP_MODEL == "flan":

    tokenizer, nlp_model = load_flan()

elif NLP_MODEL == "bart":

    tokenizer, nlp_model = load_bart()

else:
    raise ValueError()

# =====================================================
# DATA
# =====================================================

df = pd.read_csv(CSV_PATH).sample(SAMPLES)

results = []

# =====================================================
# EVAL
# =====================================================

for idx, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    image_path = row["image"]

    gt_text = str(row["text"])
    gt_expr = str(row["target"])
    gt_result = str(row["result"])

    # OCR

    if OCR_MODEL == "trocr":

        ocr_text, confidence = trocr_predict(
            ocr_processor,
            ocr_model,
            image_path
        )

    else:

        ocr_text, confidence = crnn_predict(
            ocr_model,
            charset,
            image_path
        )

    error_type = classify_error(
        gt_text,
        ocr_text
    )

    # BEFORE

    pred_expr_before = nlp_predict(
        tokenizer,
        nlp_model,
        ocr_text
    )

    try:
        pred_result_before = str(
            eval(pred_expr_before)
        )
    except:
        pred_result_before = "ERR"

    fca_before = int(
        pred_result_before == gt_result
    )

    # AFTER

    corrected_text = dictionary_correct(
        ocr_text
    )

    pred_expr_after = nlp_predict(
        tokenizer,
        nlp_model,
        corrected_text
    )

    try:
        pred_result_after = str(
            eval(pred_expr_after)
        )
    except:
        pred_result_after = "ERR"

    fca_after = int(
        pred_result_after == gt_result
    )

    if(LOG_ENABLED):

        print("\n" + "=" * 60)

        print("GT TEXT      :", gt_text)
        print("OCR TEXT     :", ocr_text)
        print("CONFIDENCE   :", round(confidence, 4))
        print("ERROR TYPE   :", error_type)

        print("NLP BEFORE   :", pred_expr_before)
        print("FCA BEFORE   :", fca_before)

        print("CORRECTED    :", corrected_text)

        print("NLP AFTER    :", pred_expr_after)
        print("FCA AFTER    :", fca_after)

    results.append({
        "gt_text": gt_text,
        "ocr_text": ocr_text,
        "confidence": confidence,
        "error_type": error_type,
        "nlp_before": pred_expr_before,
        "fca_before": fca_before,
        "corrected_text": corrected_text,
        "nlp_after": pred_expr_after,
        "fca_after": fca_after,
    })

# =====================================================
# SAVE CSV
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUT_CSV,
    index=False
)

# =====================================================
# GLOBAL STATS
# =====================================================

print("\n")
print("=" * 60)
print("GLOBAL")
print("=" * 60)

print(
    "FCA BEFORE:",
    round(
        results_df["fca_before"].mean(),
        4
    )
)

print(
    "FCA AFTER :",
    round(
        results_df["fca_after"].mean(),
        4
    )
)

# =====================================================
# ERROR TYPE STATS
# =====================================================

print("\n")
print("=" * 60)
print("BY ERROR TYPE")
print("=" * 60)

for err_type in sorted(
    results_df["error_type"].unique()
):

    part = results_df[
        results_df["error_type"] == err_type
    ]

    print()

    print(err_type)

    print(
        "count      =",
        len(part)
    )

    print(
        "before FCA =",
        round(
            part["fca_before"].mean(),
            4
        )
    )

    print(
        "after FCA  =",
        round(
            part["fca_after"].mean(),
            4
        )
    )

print("\nSaved:", OUT_CSV)
