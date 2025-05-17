import numpy as np
import torch
import torch.nn as nn
from basicts.archs.arch_zoo.stgcn_arch import STGCN

class STGCN_Step(nn.Module):
    """STGCN wrapper so it plugs into STEP.  Always uses a fixed graph."""
    def __init__(self, adj_path, num_nodes, in_dim, out_dim, hid_dim=64, Kt=3, Ks=3, act_func='glu', 
                 graph_conv_type='cheb_graph_conv', bias=True, droprate=0.5, **kwargs):
        super().__init__()
        
        # Load and register the fixed adjacency matrix
        A = np.load(adj_path)["adj"].astype(np.float32)            # (N,N)
        self.register_buffer("A_fixed", torch.from_numpy(A))       # not trainable
        
        # Projection layer for TSFormer hidden states
        # Assumes TSFormer hidden dimension is 96 (default in config)
        tsformer_dim = kwargs.get('tsformer_dim', 96)
        self.hidden_projection = nn.Linear(tsformer_dim, 1)
        self.use_hidden_states = True
        
        # Define the blocks structure for STGCN
        # Based on the standard STGCN architecture from the paper
        # The first element of blocks[0] must match the input channel dimension (1)
        # The last element of blocks[-1] must match the output dimension (out_dim)
        blocks = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [out_dim]]
        
        # Use the input sequence length for T
        # This is important as it affects the calculation of convolution layers
        T = 12  # Default input sequence length for time series forecasting
        
        # Create the STGCN model
        self.model = STGCN(
            Kt=Kt,            # Kernel size in time dimension
            Ks=Ks,            # Kernel size in spatial dimension
            blocks=blocks,    # Network architecture blocks
            T=T,              # Input sequence length
            n_vertex=num_nodes,  # Number of nodes in the graph
            act_func=act_func,   # Activation function
            graph_conv_type=graph_conv_type,  # Type of graph convolution
            gso=self.A_fixed,    # Graph shift operator (adjacency matrix)
            bias=bias,           # Whether to use bias
            droprate=droprate    # Dropout rate
        )# STEP calls backend(x, sampled_adj, hidden_states, …)
    def forward(self, x, sampled_adj=None, hidden_states=None):
        """
        Args:
            x: Input time series data, shape [B, L, N, C]
                B: Batch size
                L: Sequence length
                N: Number of nodes
                C: Number of features/channels
            sampled_adj: Sampled adjacency matrix (not used in STGCN as it uses fixed adj matrix)
            hidden_states: Hidden states from TSFormer, shape [B, N, D]
                D: Hidden dimension from TSFormer (typically 96)
        """
        # Extract only first channel from input to ensure we have [B, L, N, 1]
        if x.shape[-1] > 1:
            x_input = x[..., 0:1]  # Keep only the first channel but maintain dimension
        else:
            x_input = x
            
        # Incorporate the hidden states from TSFormer if available
        if hidden_states is not None and self.use_hidden_states:
            # Project hidden states from TSFormer's embedding dimension to 1
            # Shape goes from [B, N, D] -> [B, N, 1]
            projected_hidden = self.hidden_projection(hidden_states)
            
            # Reshape to match input dimensions [B, L, N, 1]
            # We expand the projected hidden states across the time dimension
            hidden_expanded = projected_hidden.unsqueeze(1).expand(-1, x_input.shape[1], -1, -1)
            
            # Add the projected hidden states to input features (element-wise addition)
            # This preserves the input shape while incorporating hidden state information
            x_enhanced = x_input + hidden_expanded
        else:
            x_enhanced = x_input
            
        # STGCN expects [B, C, L, N], but our input is [B, L, N, C]
        # Permute dimensions for STGCN
        x_permuted = x_enhanced.permute(0, 3, 1, 2)
        
        # Forward pass through STGCN model
        # Output shape from STGCN is [B, L, N, 1]
        output = self.model(x_permuted, None, None, None, False)
        
        # Reshape to the expected output format [B, N, L]
        return output.squeeze(-1).transpose(1, 2)  # [B, L, N] -> [B, N, L]
