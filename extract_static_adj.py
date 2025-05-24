#!/usr/bin/env python
# export_static_adj.py  –  save a fixed adjacency from TSFormer

import os, sys, pathlib, importlib
import numpy as np
import torch
import torch.nn.functional as F

# ── EDITS DONE ────────────────────────────────────────────────────────
CKPT_PATH = "tsformer_ckpt/TSFormer_METR-LA.pt"
SAVE_DIR = "/content/drive/MyDrive/Bachelor/extracted graphs"
BATCH_COUNT = 10  # Number of batches to average for representativeness
# ──────────────────────────────────────────────────────────────────────

ROOT = os.getcwd()
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "step"))

# load python config as package sub-module
cfg_mod = importlib.import_module("step.STEP_METR_LA")
cfg     = cfg_mod.CFG

# build validation data loader
from step.step_data.forecasting_dataset import ForecastingDataset
data_dir  = cfg.VAL.DATA.DIR
data_pkl  = os.path.join(data_dir, "data_in2016_out12.pkl")
index_pkl = os.path.join(data_dir, "index_in2016_out12.pkl")

val_set    = ForecastingDataset(data_pkl, index_pkl,
                                mode="valid",
                                seq_len=cfg.DATASET_INPUT_LEN)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

# build STEP & load TSFormer weights only
from step.step_arch.step import STEP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = STEP(**cfg.MODEL.PARAM).to(device)

# ── load checkpoint & pull TSFormer weights ───────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=device)

# ❶ grab the actual state-dict no matter how the file was saved
if "model_state_dict" in ckpt:
    full_state = ckpt["model_state_dict"]
elif "state_dict" in ckpt:
    full_state = ckpt["state_dict"]
else:
    full_state = ckpt

# ❷ keep ONLY the tsformer parameters; strip prefix if present
ts_state = {}
for name, value in full_state.items():
    if name.startswith("tsformer."):
        ts_state[name.replace("tsformer.", "")] = value
    elif "tsformer" not in name:
        ts_state[name] = value

# ❸ load (strict=False lets irrelevant keys pass)
missing, unexpected = model.tsformer.load_state_dict(ts_state, strict=False)
print(f"✓  Loaded {len(ts_state) - len(unexpected)}/{len(model.tsformer.state_dict())} "
      f"TSFormer parameters")
if missing:
    print(f"   ↳  {len(missing)} optional keys were missing (safe to ignore)")

model.eval()

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize containers for batch-averaged components
all_sampled_adjs = []
all_knn_adjs = []
all_prob_adjs = []
all_hidden_states = []  # For optional embedding analysis

# Process multiple batches
print(f"\n=== Processing {BATCH_COUNT} validation batches for representative graph ===")
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= BATCH_COUNT:
            break
            
        long_hist = batch[2].to(device)
        print(f"  ↳ Processing batch {i+1}/{BATCH_COUNT}")
        
        # Get graph components from the discrete graph learning module
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = model.discrete_graph_learning(
            long_hist, model.tsformer)
        
        # Get probability distribution from bernoulli parameters
        A_pred_prob = F.softmax(bernoulli_unnorm, dim=-1)[..., 0].reshape(1, 207, 207)
        
        # Extract all components
        all_sampled_adjs.append(sampled_adj[0].cpu().numpy())        # (N, N)
        all_knn_adjs.append(adj_knn[0].cpu().numpy())                # (N, N)
        all_prob_adjs.append(A_pred_prob[0].cpu().numpy())           # (N, N)
        all_hidden_states.append(hidden_states[0, :, -1, :].cpu().numpy())  # (N, d)

# Average the adjacency matrices
avg_sampled_adj = np.mean(all_sampled_adjs, axis=0)
avg_knn_adj = np.mean(all_knn_adjs, axis=0)
avg_prob_adj = np.mean(all_prob_adjs, axis=0)

# Convert probabilistic averaged sampled_adj to binary using threshold
binary_sampled_adj = (avg_sampled_adj > 0.5).astype(np.float32)

# For embeddings, we can keep the last batch's embeddings for simplicity
# Or optionally average them as well
embeddings = all_hidden_states[-1]  # Last batch embeddings

print("\n=== Graph Summary ===")
print(f"   ↳  avg_sampled_adj: {avg_sampled_adj.shape}, avg edges: {avg_sampled_adj.sum()/len(all_sampled_adjs):.1f}")
print(f"   ↳  binary_sampled_adj: {binary_sampled_adj.shape}, edges: {binary_sampled_adj.sum()}")
print(f"   ↳  avg_knn_adj: {avg_knn_adj.shape}, avg edges: {avg_knn_adj.sum()/len(all_knn_adjs):.1f}")
print(f"   ↳  avg_prob_adj: {avg_prob_adj.shape}")

# Save each component to a separate file with descriptive names
np.savez_compressed(os.path.join(SAVE_DIR, "avg_sampled_adj.npz"), 
                   adj_prob=avg_sampled_adj,
                   adj_binary=binary_sampled_adj)
np.savez_compressed(os.path.join(SAVE_DIR, "sampled_adj.npz"), adj=binary_sampled_adj)
np.savez_compressed(os.path.join(SAVE_DIR, "knn_adj.npz"), adj=avg_knn_adj)
np.savez_compressed(os.path.join(SAVE_DIR, "prob_adj.npz"), adj=avg_prob_adj)
np.savez_compressed(os.path.join(SAVE_DIR, "embeddings.npz"), embeddings=embeddings)


print(f"\n✓  Saved all graph components to {SAVE_DIR}")
print(f"   ↳  Averaged across {BATCH_COUNT} validation batches")
