from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PARSEQ_DIR = ROOT / "parseq"

cmd = [
    sys.executable,
    "train.py",

    # DATA
    "data.root_dir=../data",
    "data.augment=true",

    # CHARSET
    "model.charset_train=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "model.charset_test=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",

    # TRAIN
    "trainer.devices=1",
    "trainer.max_epochs=50",
    "trainer.val_check_interval=1.0",

    # MODEL
    "model.batch_size=32",
    "model.lr=3e-4",
    "model.weight_decay=0.05",
    "model.warmup_pct=0.1",
]

subprocess.run(cmd, cwd=PARSEQ_DIR, check=True)