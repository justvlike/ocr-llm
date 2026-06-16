import torch
from pathlib import Path
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# PATH TO CHECKPOINT
# =========================
CKPT_PATH = Path(
    r"E:\Files\PycharmProjects\ocr-llm\parseq\outputs\parseq\2026-06-12_19-41-39\checkpoints\last.ckpt"
)

assert CKPT_PATH.exists(), "Checkpoint not found"


# =========================
# LOAD CHECKPOINT
# =========================
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt["state_dict"]

# remove lightning prefix
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}


# =========================
# IMPORT MODEL ARCH
# =========================
sys.path.append(str(Path(__file__).resolve().parent.parent / "parseq"))

from strhub.models.parseq.model import PARSeq


# =========================
# IMPORTANT: use SAME config as training
# =========================
MODEL_CFG = dict(
    num_tokens=62,
    max_label_length=32,
    img_size=(32, 128),
    patch_size=(4, 8),
    embed_dim=384,
    enc_num_heads=6,
    enc_mlp_ratio=4,
    enc_depth=12,
    dec_num_heads=12,
    dec_mlp_ratio=4,
    dec_depth=1,
    decode_ar=True,
    refine_iters=1,
    dropout=0.1,
)


def build_model():
    model = PARSeq(**MODEL_CFG)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] missing keys: {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    model = model.to(DEVICE)
    model.eval()
    return model


# =========================
# INFERENCE FUNCTION
# =========================
@torch.no_grad()
def predict(model, images):
    """
    images: torch.Tensor [B, C, H, W]
    """
    images = images.to(DEVICE)
    out = model(images)
    return out