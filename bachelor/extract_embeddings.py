import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# 1) point to your config
from configs.TSFormer_METR_LA import CFG

# 2) imports from your STEP fork
from basicts.archs.tsformer import TSFormer
from basicts.data import PretrainingDataset  # the one your config uses

# 3) device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 4) build TSFormer with exactly your pre-training params
tsf = TSFormer(
    patch_size=CFG.MODEL.PARAM["patch_size"],
    in_channel=CFG.MODEL.PARAM["in_channel"],
    embed_dim=CFG.MODEL.PARAM["embed_dim"],
    num_heads=CFG.MODEL.PARAM["num_heads"],
    mlp_ratio=CFG.MODEL.PARAM["mlp_ratio"],
    dropout=CFG.MODEL.PARAM["dropout"],
    num_token=int(CFG.MODEL.PARAM["num_token"]),
    mask_ratio=CFG.MODEL.PARAM["mask_ratio"],
    encoder_depth=CFG.MODEL.PARAM["encoder_depth"],
    decoder_depth=CFG.MODEL.PARAM["decoder_depth"],
    mode=CFG.MODEL.PARAM["mode"],
).to(device)

# 5) load your pretrained weights
ckpt = torch.load("tsformer_ckpt/TSFormer_METR-LA.pt", map_location=device)
tsf.load_state_dict(ckpt["model_state_dict"])
tsf.eval()

# 6) dataset just for **long history** (input_len = 288*7)
ds = PretrainingDataset(
    root=CFG.TRAIN.DATA.DIR,
    in_len=CFG.DATASET_INPUT_LEN,
    out_len=CFG.DATASET_OUTPUT_LEN,
    split="all"
)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=CFG.TRAIN.DATA.NUM_WORKERS)

# 7) run encoder and grab the last‐patch embedding
embs = []
with torch.no_grad():
    for long_hist, _ in loader:        # PretrainingDataset returns (long_hist, target)
        # long_hist: [B=1, T=288*7, N, C=1]
        B, T, N, C = long_hist.shape
        P = int(CFG.MODEL.PARAM["num_token"])
        L = CFG.MODEL.PARAM["patch_size"]
        x = long_hist.view(B, P, L, C).to(device)    # → [1, P, L, 1]
        H = tsf.encoder(x)                           # H: [1, P, N, d]
        h_last = H[:, -1, :, :]                      # [1, N, d]
        embs.append(h_last.squeeze(0).cpu().numpy())

# stack and average (here all “all” means train+val+test; you can split afterward)
emb_matrix = np.stack(embs, axis=0).mean(axis=0)  # → (N=207, d=96)

# 8) save
os.makedirs("precomputed", exist_ok=True)
np.save("precomputed/segment_embeddings_METR-LA.npy", emb_matrix)
print("Saved to precomputed/segment_embeddings_METR-LA.npy:", emb_matrix.shape)
