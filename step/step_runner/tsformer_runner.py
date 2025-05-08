import torch
import os
import numpy as np

from easytorch.utils.dist import master_only
from basicts.data.registry import SCALER_REGISTRY
from basicts.runners import BaseTimeSeriesForecastingRunner


class TSFormerRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        checkpoint = torch.load(cfg.TSFORMER_CKPT_PATH, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs):
        _, history_data = data  # Ignore future_data
        history_data = self.to_running_device(history_data)
        history_data = self.select_input_features(history_data)

        # Get embeddings from TSFormer
        hidden_states_full = self.model(history_data=history_data)
        return hidden_states_full


    @torch.no_grad()
    def save_embeddings(self, save_path):
        self.model.eval()
        all_embeddings = []

        for _, data in enumerate(self.test_data_loader):
            embeddings = self.forward(data, train=False)
            all_embeddings.append(embeddings.cpu())

        # Concatenate all and save
        all_embeddings = torch.cat(all_embeddings, dim=0)  # [B_total, N, L, D]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.save(save_path, all_embeddings.numpy())

   