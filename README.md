## ocr-llm
OCR project

---

# Process
## 0. Setup
- using Py 3.12\
- installed Libs: ```pip install pandas numpy matplotlib seaborn scikit-learn opencv-python albumentations optuna fastapi uvicorn transformers datasets sentence-transformers faiss-cpu```\
- installed PyTorch GPU: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

## 1. Data collection
- downloaded and parsed EMNIST, created emnist.csv format image -> text (first 10000 imgs)\
- downloaded and parsed IAM (ascii, lines), created iam.csv format image -> text\
- combined data into full_dataset.csv

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
- created base CRNN + CTC Loss train_baseline.py
- attached train loop
- saved into crnn_baseline.pth

## 5. Metrics
- сalculate CER, WER in crnn_validate_baseline.py
- зrovided random visualization