# extract_tsformer_embeddings.py
import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader

# 1) point these at your files:
TSFORMER_CKPT = "tsformer_ckpt/TSFormer_METR-LA.pt"
DATA_PKL        = "datasets/METR-LA/processed_data.pkl"
INDEX_PKL       = "datasets/METR-LA/index.pkl"

# 2) import your code
from step.tsformer import TSFormer
from step.forecasting_dataset import ForecastingDataset

# 3) copy your tsformer_args from STEP_METR-LA.py
tsformer_args = {
    "patch_size":    12,
    "in_channel":     1,
    "embed_dim":     96,
    "num_heads":      4,
    "mlp_ratio":      4,
    "dropout":       0.1,
    "num_token":   (288 * 7) // 12,
    "mask_ratio":   0.75,
    "encoder_depth": 4,
    "decoder_depth": 1,
    "mode":      "forecasting"
}

# 4) instantiate and load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TSFormer(**tsformer_args).to(device)
ckpt = torch.load(TSFORMER_CKPT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 5) build a DataLoader over just the long-history tensor
#    ForecastingDataset.__getitem__ returns (future, history, long_history)
ds = ForecastingDataset(data_file_path=DATA_PKL,
                        index_file_path=INDEX_PKL,
                        mode="test",
                        seq_len=288*7)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

# 6) run once through and collect
all_embs = []
with torch.no_grad():
    for future, history, long_hist in loader:
        # long_hist: [B, L*P, N, C]
        long_hist = long_hist.to(device)
        # TSFormer expects [B, N, C, L*P]
        H = model(long_hist.permute(0,2,3,1))  
        # H: [B, N, P, d]
        all_embs.append(H.cpu().numpy())

all_embs = np.concatenate(all_embs, axis=0)  # [#samples, N, P, d]
print("Collected embeddings:", all_embs.shape)

# 7) save
np.save("tsformer_embeddings.npy", all_embs)
print("Saved to tsformer_embeddings.npy")
