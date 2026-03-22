import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
df = pd.read_csv(csv_path)

print(f"Initial dataset size: {len(df)}")

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)
df = df[df["text"].str.lower() != "nan"]
df = df[df["text"].str.strip() != ""]

print(f"After cleaning: {len(df)}")

# Visualize random data
sample = df.sample(5)
plt.figure(figsize=(12, 4))

for i, (_, row) in enumerate(sample.iterrows()):
    img_path = os.path.join(BASE_DIR, row["image_path"])
    img = Image.open(img_path)

    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(row["text"][:10])
    plt.axis("off")

plt.show()

# Data length
df["text_length"] = df["text"].apply(lambda x: len(str(x)))

print("\nText length:=")
print(df["text_length"].describe())

plt.figure()
df["text_length"].hist(bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Length")
plt.ylabel("Count")
plt.show()

# Symbol balance
all_text = "".join(df["text"].astype(str).values)
char_counts = Counter(all_text)

print("\nMost common chars:")
print(char_counts.most_common(20))

chars, counts = zip(*char_counts.most_common(20))

plt.figure()
plt.bar(chars, counts)
plt.title("Top 20 Characters")
plt.show()

# Hard (longest) cases check
threshold = df["text_length"].quantile(0.95)
long_texts = df[df["text_length"] > threshold]

if len(long_texts) > 0:
    sample = long_texts.sample(min(5, len(long_texts)))

    plt.figure(figsize=(12, 4))

    for i, (_, row) in enumerate(sample.iterrows()):
        img_path = os.path.join(BASE_DIR, row["image_path"])
        img = Image.open(img_path)

        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"len={len(row['text'])}")
        plt.axis("off")

    plt.show()

# Short cases
short_texts = df[df["text_length"] < 3]

if len(short_texts) > 0:
    sample = short_texts.sample(min(5, len(short_texts)))

    plt.figure(figsize=(12, 4))

    for i, (_, row) in enumerate(sample.iterrows()):
        img_path = os.path.join(BASE_DIR, row["image_path"])
        img = Image.open(img_path)

        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(row["text"])
        plt.axis("off")

    plt.show()

print("\nEDA DONE")