from PIL import Image
import torch

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    r"E:\Files\PycharmProjects\ocr-llm\checkpoints\trocr_finetuned"
)

image = Image.open(
    r"dataset/test/images/0.png"
).convert("RGB")

pixel_values = processor(
    image,
    return_tensors="pt"
).pixel_values

outputs = model.generate(
    pixel_values,
    return_dict_in_generate=True,
    output_scores=True,
)

print(type(outputs))
print(hasattr(outputs, "scores"))
print(len(outputs.scores))