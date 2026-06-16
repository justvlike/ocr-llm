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

from jiwer import wer
import editdistance

from model_crnn import CRNN


# =====================================================
# CONFIG
# =====================================================

OCR_MODEL = "crnn"      # trocr | crnn
NLP_MODEL = "bart"      # flan | bart

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_PATH = "dataset/test/dataset.csv"

TROCR_PATH = r"E:\Files\PycharmProjects\ocr-llm\checkpoints\trocr_finetuned"
CRNN_PATH = r"E:\Files\PycharmProjects\ocr-llm\weights\crnn_best.pt"

FLAN_PATH = "flan_nlp.pt"
BART_PATH = "bart_nlp.pt"


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

    generated_ids = model.generate(
        pixel_values
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


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
        print("SHAPE", logits.shape)

    text = ctc_decode(
        logits,
        charset
    )

    return text.strip()


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
# NLP INFERENCE
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
# CER
# =====================================================

def cer(gt, pred):

    if len(gt) == 0:
        return 0

    return editdistance.eval(
        gt,
        pred
    ) / len(gt)


# =====================================================
# LOAD MODELS
# =====================================================

print("Loading OCR...")

if OCR_MODEL == "trocr":

    ocr_processor, ocr_model = load_trocr()

elif OCR_MODEL == "crnn":

    ocr_model, charset = load_crnn()

else:

    raise ValueError("Unknown OCR_MODEL")


print("Loading NLP...")

if NLP_MODEL == "flan":

    tokenizer, nlp_model = load_flan()

elif NLP_MODEL == "bart":

    tokenizer, nlp_model = load_bart()

else:

    raise ValueError("Unknown NLP_MODEL")


# =====================================================
# DATA
# =====================================================

df = pd.read_csv(CSV_PATH)
df = df.head(100)

total = len(df)

ocr_cer = 0
ocr_wer = 0

parse_correct = 0
fca_correct = 0


# =====================================================
# EVALUATION
# =====================================================

for _, row in tqdm(
    df.iterrows(),
    total=total
):

    image_path = row["image"]

    gt_text = str(row["text"])
    gt_expr = str(row["target"])
    gt_result = str(row["result"])

    # OCR

    if OCR_MODEL == "trocr":

        ocr_text = trocr_predict(
            ocr_processor,
            ocr_model,
            image_path
        )
        print("ocr_text", ocr_text)

    else:

        ocr_text = crnn_predict(
            ocr_model,
            charset,
            image_path
        )
        print("ocr_text", ocr_text)

    # OCR metrics

    ocr_cer += cer(
        gt_text,
        ocr_text
    )

    ocr_wer += wer(
        gt_text,
        ocr_text
    )

    # NLP

    pred_expr = nlp_predict(
        tokenizer,
        nlp_model,
        ocr_text
    )
    print("pred_exprt", pred_expr)

    if pred_expr == gt_expr:

        parse_correct += 1

    try:

        pred_result = str(
            eval(pred_expr)
        )
        print("pred_expr", pred_expr, "pred_result:", pred_result)

    except:

        pred_result = "ERR"

    if pred_result == gt_result:

        fca_correct += 1


# =====================================================
# RESULTS
# =====================================================

print("\n========================")
print("PIPELINE RESULTS")
print("========================")

print({
    "ocr": OCR_MODEL,
    "nlp": NLP_MODEL,
    "samples": total,
    "ocr_cer": round(ocr_cer / total, 4),
    "ocr_wer": round(ocr_wer / total, 4),
    "parse_accuracy": round(parse_correct / total, 4),
    "FCA": round(fca_correct / total, 4),
})