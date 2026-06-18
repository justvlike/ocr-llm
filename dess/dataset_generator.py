import os
import csv
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =====================================================
# CONFIG
# =====================================================

MODE = "train"  # train | test | both

BASE_DIR = "dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train_critical")
TEST_DIR = os.path.join(BASE_DIR, "test_trocr")

TRAIN_IMG = os.path.join(TRAIN_DIR, "images")
TEST_IMG = os.path.join(TEST_DIR, "images")

FONT_DIR = os.path.join(BASE_DIR, "fonts")

IMG_W = 256
IMG_H = 96   # 🔥 increased height for CRNN stability

N_TRAIN = 9000
N_TEST = 1000

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TEST_IMG, exist_ok=True)

# =====================================================
# DATA
# =====================================================

NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

OPS = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "divided by": "/"
}

WORDS = list(NUMBERS.keys())
OPS_WORDS = list(OPS.keys())

# =====================================================
# FONTS (ROUND-ROBIN)
# =====================================================

def load_fonts():
    fonts = []
    for f in sorted(os.listdir(FONT_DIR)):
        if f.lower().endswith(".ttf") or f.lower().endswith(".otf"):
            fonts.append(os.path.join(FONT_DIR, f))

    if not fonts:
        raise ValueError("No fonts found")

    print(f"Loaded fonts: {len(fonts)}")
    return fonts


FONTS = load_fonts()

# =====================================================
# EXPRESSION
# =====================================================

def generate_expression():
    a_word = random.choice(WORDS)
    b_word = random.choice(WORDS)
    op_word = random.choice(OPS_WORDS)

    a = int(NUMBERS[a_word])
    b = int(NUMBERS[b_word])

    expr_text = f"{a_word} {op_word} {b_word}"
    expr = f"{a}{OPS[op_word]}{b}"

    if OPS[op_word] == "+":
        result = a + b
    elif OPS[op_word] == "-":
        result = a - b
    elif OPS[op_word] == "*":
        result = a * b
    else:
        result = a // b if b != 0 else 0

    return expr_text, expr, str(result)

# =====================================================
# AUGMENTATION (SAFE)
# =====================================================

def add_noise(img, intensity=5):
    arr = np.array(img).astype(np.int16)

    noise = np.random.randint(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def transform(img, mode="train"):
    if mode != "train":
        return img

    if random.random() < 0.5:
        img = img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6))
        )

    if random.random() < 0.6:
        img = img.rotate(
            random.uniform(-2, 2),
            fillcolor=255,
            resample=Image.BILINEAR
        )

    return img

# =====================================================
# RENDER (IMPORTANT FIXES HERE)
# =====================================================

def render(text, font_path, mode="train"):

    # 🔥 STRICT WHITE BACKGROUND
    img = Image.new("L", (IMG_W, IMG_H), 255)
    draw = ImageDraw.Draw(img)

    # 🔥 BIGGER FONT (CRNN-friendly)
    font_size = random.randint(28, 40)

    font = ImageFont.truetype(font_path, font_size)

    # 🔥 CENTERED TEXT (important stability boost)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = max(5, (IMG_W - text_width) // 2)
    y = max(5, (IMG_H - text_height) // 2)

    draw.text((x, y), text, font=font, fill=0)

    img = transform(img, mode=mode)

    if mode == "train":
        img = add_noise(img, intensity=5)
    else:
        img = add_noise(img, intensity=2)

    return img

# =====================================================
# SPLIT GENERATION
# =====================================================

def generate_split(n, img_dir, csv_path, mode, fonts):

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text", "target", "result"])

        for i in range(n):

            text, expr, result = generate_expression()

            font_path = fonts[i % len(fonts)]  # 🔥 round-robin fonts

            img = render(text, font_path, mode)

            path = os.path.join(img_dir, f"{i}.png")
            img.save(path)

            writer.writerow([
                path.replace("\\", "/"),
                text,
                expr,
                result
            ])

            if i % 500 == 0:
                print(f"[{mode}] {i}/{n} font={os.path.basename(font_path)}")

# =====================================================
# RUN
# =====================================================

if MODE in ["train", "both"]:
    print("\nGenerating TRAIN...")
    generate_split(
        N_TRAIN,
        TRAIN_IMG,
        os.path.join(TRAIN_DIR, "dataset.csv"),
        "train",
        FONTS
    )

if MODE in ["test", "both"]:
    print("\nGenerating TEST...")
    generate_split(
        N_TEST,
        TEST_IMG,
        os.path.join(TEST_DIR, "dataset.csv"),
        "test",
        FONTS
    )

print("\nDONE")