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
        
        # Define the blocks structure for STGCN
        # Based on the standard STGCN architecture from the paper
        blocks = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [out_dim]]
        T = 12  # Default input sequence length for time series forecasting
        
        # Create the STGCN model
        self.model = STGCN(
            Kt=Kt,
            Ks=Ks,
            blocks=blocks,
            T=T,
            n_vertex=num_nodes,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            gso=self.A_fixed,
            bias=bias,
            droprate=droprate
        )

    # STEP calls backend(x, sampled_adj, hidden_states, …)
    def forward(self, x, sampled_adj=None, hidden_states=None):
        # Incorporate hidden states if provided (as additional features)
        if hidden_states is not None:
            # Reshape hidden_states to match the expected input shape
            hidden_reshaped = hidden_states.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
            # Concatenate with the input data along the channel dimension
            x = torch.cat([x, hidden_reshaped], dim=-1)
        
        # Use the fixed adjacency matrix rather than the sampled_adj
        # Output shape needs to be (B,N,L)
        output = self.model(x, None, None, None, False)
        return output.squeeze(-1)  # Remove the channel dimension
