#!/usr/bin/env python
# export_static_adj.py  –  save a fixed adjacency from TSFormer

import os, sys, pathlib, importlib
import numpy as np
import torch
import torch.nn.functional as F  # Add missing import for F.softmax

# ── EDITS DONE ────────────────────────────────────────────────────────
CKPT_PATH = "tsformer_ckpt/TSFormer_METR-LA.pt"      # ① correct path
SAVE_DIR = "/content/drive/MyDrive/Bachelor/extracted graphs"
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
    print(f"   ↳  {len(missing)} optional keys were missing (safe to ignore)")  # Fix missing f-string
model.eval()

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# sample adjacency once
with torch.no_grad():
    bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = model.discrete_graph_learning(
        long_hist.to(device), model.tsformer)
    
    # Get probability distribution from bernoulli parameters
    A_pred_prob = F.softmax(bernoulli_unnorm, dim=-1)[..., 0].reshape(1, 207, 207)
    
    # Extract all components
    A_sampled = sampled_adj[0].cpu().numpy()        # (N, N)
    A_knn = adj_knn[0].cpu().numpy()                # (N, N)
    A_prob = A_pred_prob[0].cpu().numpy()           # (N, N)
    
    # For reference, also save embeddings (last patch only, to keep file size reasonable)
    embeddings = hidden_states[0, :, -1, :].cpu().numpy()  # (N, d)

print(f"   ↳  adj_sampled: {A_sampled.shape}, edges: {A_sampled.sum()}")
print(f"   ↳  adj_knn: {A_knn.shape}, edges: {A_knn.sum()}")
print(f"   ↳  adj_prob: {A_prob.shape}")
print(f"   ↳  embeddings: {embeddings.shape}")

# Save each component to a separate file with descriptive names
np.savez_compressed(os.path.join(SAVE_DIR, "sampled_adj.npz"), adj=A_sampled)
np.savez_compressed(os.path.join(SAVE_DIR, "knn_adj.npz"), adj=A_knn)
np.savez_compressed(os.path.join(SAVE_DIR, "prob_adj.npz"), adj=A_prob)
np.savez_compressed(os.path.join(SAVE_DIR, "embeddings.npz"), embeddings=embeddings)

print(f"✓  Saved all graph components to {SAVE_DIR}")
