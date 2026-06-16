import os
import zipfile
import glob
import xml.etree.ElementTree as ET
import pandas as pd


# =========================
# CONFIG
# =========================
ZIP_PATH = "data/crohme.zip"
OUT_DIR = "data/crohme"
CSV_OUT = "data/crohme.csv"


# =========================
# UNZIP
# =========================
def unzip():
    os.makedirs(OUT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(OUT_DIR)

    print("Extracted CROHME")


# =========================
# PARSE INKML
# =========================
def parse_inkml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # CROHME uses LaTeX in <annotation type="truth">
    for ann in root.findall(".//annotation"):
        if ann.attrib.get("type") == "truth":
            return ann.text

    return None


# =========================
# BUILD CSV
# =========================
def build_csv():
    inkml_files = glob.glob(os.path.join(OUT_DIR, "**/*.inkml"), recursive=True)

    rows = []

    for f in inkml_files:
        expr = parse_inkml(f)

        if expr is None:
            continue

        rows.append({
            "file": f,
            "expression": expr
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    print(f"Saved: {CSV_OUT}, samples={len(df)}")


# =========================
if __name__ == "__main__":
    unzip()
    build_csv()