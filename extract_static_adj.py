#!/usr/bin/env python
# export_static_adj.py  –  save a fixed adjacency from TSFormer

import os, sys, pathlib, importlib
import numpy as np
import torch

# ── EDITS DONE ────────────────────────────────────────────────────────
CKPT_PATH = "tsformer_ckpt/TSFormer_METR-LA.pt"      # ① correct path
SAVE_PATH = "/content/drive/MyDrive/Bachelor/static_adj.npz"
# ──────────────────────────────────────────────────────────────────────

ROOT = os.getcwd()          # repo root (./STEP)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "step"))          # ④ ensure package

# load python config as package sub-module
cfg_mod = importlib.import_module("step.STEP_METR_LA")
cfg     = cfg_mod.CFG

# build ONE val batch
from step.step_data.forecasting_dataset import ForecastingDataset  # ③
data_dir  = cfg.VAL.DATA.DIR
data_pkl  = os.path.join(data_dir, "data_in2016_out12.pkl")
index_pkl = os.path.join(data_dir, "index_in2016_out12.pkl")

val_set    = ForecastingDataset(data_pkl, index_pkl,
                                mode="valid",
                                seq_len=cfg.DATASET_INPUT_LEN)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)
long_hist  = next(iter(val_loader))[2]         # (1, L, N, C)

# build STEP & load TSFormer weights only
from step.step_arch.step import STEP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = STEP(**cfg.MODEL.PARAM).to(device)

# ── load checkpoint & pull TSFormer weights ───────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=device)

# ❶ grab the actual state-dict no matter how the file was saved
if "model_state_dict" in ckpt:          # training snapshot (your case)
    full_state = ckpt["model_state_dict"]
elif "state_dict" in ckpt:              # wrapped state_dict
    full_state = ckpt["state_dict"]
else:                                   # raw state_dict
    full_state = ckpt                   # already the weights

# ❷ keep ONLY the tsformer parameters; strip prefix if present
ts_state = {}
for name, value in full_state.items():
    if name.startswith("tsformer."):
        ts_state[name.replace("tsformer.", "")] = value
    elif "tsformer" not in name:        # pure-TSFormer checkpoint
        ts_state[name] = value

# ❸ load (strict=False lets irrelevant keys pass)
missing, unexpected = model.tsformer.load_state_dict(ts_state, strict=False)
print(f"✓  Loaded {len(ts_state) - len(unexpected)}/{len(model.tsformer.state_dict())} "
      f"TSFormer parameters")
if missing:
    print("   ↳  {len(missing)} optional keys were missing (safe to ignore)")
model.eval()



# sample adjacency once
with torch.no_grad():
    _, _, _, sampled_adj = model.discrete_graph_learning(
        long_hist.to(device), model.tsformer)

A = sampled_adj[0].cpu().numpy()        # (N, N)

np.savez_compressed(SAVE_PATH, adj=A)
print(f"✅  Saved {SAVE_PATH} | shape {A.shape}")
