#!/usr/bin/env python
"""
Extract TSFormer long-history embeddings for the METR-LA **test** split
and save them to a .npy file.

Run from a Colab notebook **after** mounting Drive:

    !python extract_tsformer_embeddings.py \
        --out_dir "/content/drive/MyDrive/tsformer_emb" \
        --batch_size 8 \
        --max_samples 512          # optional sanity-check
"""

import os, argparse, pickle, time          ### NEW ###
import numpy as np
import torch
from torch.utils.data import DataLoader

# ╭────────────────── argparse ─────────────────────────╮
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",      default="/content/drive/MyDrive/tsformer_emb")
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--max_samples",  type=int, default=None,      ### NEW ###
                    help="Only embed the first N test samples "
                         "(handy for quick checks).")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
# ╰──────────────────────────────────────────────────────╯

# 1) --- paths to your files ---------------------------------------------------
TSFORMER_CKPT = "tsformer_ckpt/TSFormer_METR-LA.pt"
DATA_PKL      = "datasets/METR-LA/data_in2016_out12.pkl"
INDEX_PKL     = "datasets/METR-LA/index_in2016_out12.pkl"

# 2) --- import project modules ----------------------------------------------
from step.step_arch.tsformer.tsformer import TSFormer
from step.step_data.forecasting_dataset import ForecastingDataset

# 3) --- hyper-parameters copied from STEP_METR-LA.py -------------------------
tsformer_args = dict(
    patch_size     = 12,
    in_channel     = 1,
    embed_dim      = 96,
    num_heads      = 4,
    mlp_ratio      = 4,
    dropout        = 0.1,
    num_token      = (288 * 7) // 12,
    mask_ratio     = 0.75,
    encoder_depth  = 4,
    decoder_depth  = 1,
    mode           = "forecasting",
)

# 4) --- model ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TSFormer(**tsformer_args).to(device)
ckpt   = torch.load(TSFORMER_CKPT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"[INFO] Model loaded on {device} — params: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# 5) --- DataLoader over *long-history* tensors -------------------------------
ds = ForecastingDataset(
    data_file_path   = DATA_PKL,
    index_file_path  = INDEX_PKL,
    mode             = "test",
    seq_len          = 288 * 7
)
if args.max_samples is not None:                    ### NEW ###
    ds.index = ds.index[:args.max_samples]
    print(f"[INFO] Truncated dataset to first {len(ds)} samples")

loader = DataLoader(ds,
                    batch_size = args.batch_size,
                    shuffle    = False,
                    num_workers= 2,
                    pin_memory = True)

print(f"[INFO] DataLoader - {len(ds)} samples "
      f"→ {len(loader)} batches of size {args.batch_size}")

# 6) --- forward pass & collect embeddings ------------------------------------
all_embs = []
start = time.time()                                ### NEW ###
with torch.no_grad():
    for step, (_, _, long_hist) in enumerate(loader, 1):
        long_hist = long_hist[..., :1].to(device)   # [B, L*P, N, 1]
        H = model(long_hist)                       # → [B, N, P, d]
        all_embs.append(H.cpu().numpy())

        # quick progress report every 50 batches
        if step % 50 == 0 or step == len(loader):  ### NEW ###
            elapsed = time.time() - start
            done    = step * args.batch_size
            print(f"  batch {step:>4}/{len(loader)} — "
                  f"samples {done}/{len(ds)} — "
                  f"{elapsed:.1f}s elapsed")

all_embs = np.concatenate(all_embs, axis=0)        # [S, N, P, d]
print("[INFO] Collected embeddings shape:", all_embs.shape,
      "≈ {:.2f} GB".format(all_embs.nbytes / 1e9)) ### NEW ###

# 7) --- save -----------------------------------------------------------------
out_path = os.path.join(args.out_dir, "tsformer_embeddings.npy")
np.save(out_path, all_embs)
print("[INFO] Saved to:", out_path)
