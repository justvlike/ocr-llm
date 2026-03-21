## ocr-llm
OCR project

---

# Process
## 0. Setup
Py 3.12\
Libs: ```pip install pandas numpy matplotlib seaborn scikit-learn opencv-python albumentations optuna fastapi uvicorn transformers datasets sentence-transformers faiss-cpu```\
PyTorch GPU: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

## 1. Data collection
Downloaded and parsed EMNIST, created emnist.csv format image -> text (first 10000 imgs)\
Downloaded and parsed IAM (ascii, lines), created iam.csv format image -> text\
Combined data into full_dataset.csv

## 2. Data preprocessing