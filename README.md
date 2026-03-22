## ocr-llm
OCR project

---

# Process
## 0. Setup
Using Py 3.12\
Installed Libs: ```pip install pandas numpy matplotlib seaborn scikit-learn opencv-python albumentations optuna fastapi uvicorn transformers datasets sentence-transformers faiss-cpu```\
Installed PyTorch GPU: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

## 1. Data collection
Downloaded and parsed EMNIST, created emnist.csv format image -> text (first 10000 imgs)\
Downloaded and parsed IAM (ascii, lines), created iam.csv format image -> text\
Combined data into full_dataset.csv

## 2. Data preprocessing
Performed:
- data resized and grayscaled
- noise reduction
- data binarized and normalized
- data split train/val/test
- leakage check

## 3. Exploratory Data Analysis (EDA)
- visualized random data for manual check
- built scale to analyze data(text) length distribution
- built scale to analyze most used characters
- visualized hardest cases (outstanding text length)
- visualized simplest cases (minimal text length)

## 4. Baseline model