from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT / "data" / "IAM" / "processed" / "iam.csv"

TRAIN_CSV = ROOT / "data" / "IAM" / "processed" / "train.csv"
VAL_CSV = ROOT / "data" / "IAM" / "processed" / "val.csv"
TEST_CSV = ROOT / "data" / "IAM" / "processed" / "test.csv"

SEED = 42

df = pd.read_csv(INPUT_CSV)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    shuffle=True,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True,
)

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

print(f"\nSaved:")
print(TRAIN_CSV)
print(VAL_CSV)
print(TEST_CSV)