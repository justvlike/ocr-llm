from pathlib import Path
import io

import lmdb
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT / "data/IAM/processed/train.csv"
VAL_CSV = ROOT / "data/IAM/processed/val.csv"
TEST_CSV = ROOT / "data/IAM/processed/test.csv"

OUT_ROOT = ROOT / "data"


def create_lmdb(csv_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    env = lmdb.open(
        str(out_dir),
        map_size=1024 ** 4
    )

    with env.begin(write=True) as txn:

        for idx, row in enumerate(df.itertuples(), start=1):

            image_path = ROOT / str(row.image_path).replace("\\", "/")

            with open(image_path, "rb") as f:
                image_bin = f.read()

            label = str(row.text).replace("|", " ")

            txn.put(
                f"image-{idx:09d}".encode(),
                image_bin
            )

            txn.put(
                f"label-{idx:09d}".encode(),
                label.encode("utf-8")
            )

            if idx % 1000 == 0:
                print(out_dir.name, idx)

        txn.put(
            b"num-samples",
            str(len(df)).encode()
        )

    env.close()

    print(
        f"{out_dir} created "
        f"({len(df)} samples)"
    )


def main():

    create_lmdb(
        TRAIN_CSV,
        OUT_ROOT / "train" / "real" / "IAM"
    )

    create_lmdb(
        VAL_CSV,
        OUT_ROOT / "val"
    )

    create_lmdb(
        TEST_CSV,
        OUT_ROOT / "test"
    )


if __name__ == "__main__":
    main()