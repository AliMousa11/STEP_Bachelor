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
        )
          # STEP calls backend(x, sampled_adj, hidden_states, …)
    def forward(self, x, sampled_adj=None, hidden_states=None):
        """
        Args:
            x: Input time series data, shape [B, L, N, C] or possibly [B, N, C, L]
                B: Batch size
                L: Sequence length (typically 12)
                N: Number of nodes (typically 207)
                C: Number of features/channels (typically 1)
            sampled_adj: Sampled adjacency matrix (not used in STGCN as it uses fixed adj matrix)
            hidden_states: Hidden states from TSFormer, shape [B, N, D]
                D: Hidden dimension from TSFormer (typically 96)
        """
        # Debug print of original input shape
        print(f"DEBUG - Original input shape: {x.shape}")
        
        # Based on the runtime error, the input shape is [B, N, C, L] = [32, 207, 1, 12]
        # STGCN expects [B, C, L, N] = [32, 1, 12, 207]
        
        # Extract only first channel from input if needed (to ensure C=1)
        if x.shape[-1] > 1:
            x_input = x[..., 0:1]  # Keep only the first channel but maintain dimension
        else:
            x_input = x
            
        # Incorporate the hidden states from TSFormer if available
        if hidden_states is not None and self.use_hidden_states:
            print("DEBUG - Using hidden states from TSFormer")
            # Project hidden states from TSFormer's embedding dimension to 1
            # Shape goes from [B, N, D] -> [B, N, 1]
            projected_hidden = self.hidden_projection(hidden_states)
            
            # Reshape to match input dimensions
            # For [B, N, C, L] input, expand hidden across L dimension
            if x_input.shape[1] == 207:  # If input is [B, N, C, L]
                hidden_expanded = projected_hidden.unsqueeze(-1).expand(-1, -1, -1, x_input.shape[-1])
                print(f"DEBUG - Expanded hidden states to: {hidden_expanded.shape}")
            else:  # Default case for [B, L, N, C] input
                hidden_expanded = projected_hidden.unsqueeze(1).expand(-1, x_input.shape[1], -1, -1)
                print(f"DEBUG - Expanded hidden states to: {hidden_expanded.shape}")
            
            # Add the projected hidden states to input features
            x_enhanced = x_input + hidden_expanded
        else:
            print("DEBUG - Not using hidden states")
            x_enhanced = x_input
        
        print(f"DEBUG - Enhanced input shape: {x_enhanced.shape}")
        
        # Apply the correct permutation based on the input shape
        try:
            if x_enhanced.shape[1] == 207:  # Input is [B, N, C, L]
                # Need to permute [B, N, C, L] -> [B, C, L, N]
                x_permuted = x_enhanced.permute(0, 2, 3, 1)
                print(f"DEBUG - Permuted from [B, N, C, L] to [B, C, L, N]: {x_permuted.shape}")
            else:
                # Default case: input is [B, L, N, C]
                x_permuted = x_enhanced.permute(0, 3, 1, 2)
                print(f"DEBUG - Permuted from [B, L, N, C] to [B, C, L, N]: {x_permuted.shape}")
              # Forward pass through STGCN model
            output = self.model(x_permuted, None, None, None, False)
            print(f"DEBUG - STGCN output shape: {output.shape}")
              # Process the output to get expected format [B, L, N, C]
            if output.dim() == 4:
                # Transform from [B, C, L, N] to [B, L, N, C]
                print(f"DEBUG - Transforming output from [B, C, L, N] to [B, L, N, C]")
                result = output.permute(0, 2, 3, 1)
                print(f"DEBUG - After permutation shape: {result.shape}")
            elif output.dim() == 3:
                # Handle 3D output [B, L, N] by adding channel dimension
                print(f"DEBUG - Adding channel dimension to 3D output: {output.shape}")
                result = output.unsqueeze(-1)  # Add channel dimension at the end
                print(f"DEBUG - After adding channel dimension: {result.shape}")
            else:
                # In case of unexpected output format, preserve as is
                print(f"DEBUG - Unexpected output dimensions: {output.dim()}")
                result = output
            
            print(f"DEBUG - Final result shape: {result.shape}")
            return result
        
        except Exception as e:
            print(f"DEBUG - Error in model forward pass: {e}")
            # Try a fallback approach
            print(f"DEBUG - Attempting fallback with alternative permutation")
            
            # Try different permutation patterns as a last resort
            try:                # Maybe input is already in the form STGCN expects?
                output = self.model(x_enhanced, None, None, None, False)                # Apply the same transformation as main path
                if output.dim() == 4:
                    # Transform from [B, C, L, N] to [B, L, N, C]
                    result = output.permute(0, 2, 3, 1)
                elif output.dim() == 3:
                    # Handle 3D output by adding channel dimension
                    result = output.unsqueeze(-1)  # Add channel dimension at the end
                else:
                    result = output
                print(f"DEBUG - Fallback succeeded with direct input. Result shape: {result.shape}")
                return result
            except:
                # Last try with a specific permutation
                if x_enhanced.shape[1] == 207:  # [B, N, C, L]
                    x_permuted = x_enhanced.permute(0, 2, 3, 1)  # -> [B, C, L, N]
                else:
                    x_permuted = x_enhanced.permute(0, 3, 1, 2)  # -> [B, C, L, N]
                output = self.model(x_permuted, None, None, None, False)                # Use the same correct transformation as the main path
                if output.dim() == 4:
                    # Transform from [B, C, L, N] to [B, L, N, C]
                    result = output.permute(0, 2, 3, 1)
                elif output.dim() == 3:
                    # Handle 3D output by adding channel dimension
                    result = output.unsqueeze(-1)  # Add channel dimension at the end
                else:
                    result = output
                print(f"DEBUG - Last resort succeeded. Result shape: {result.shape}")
                return result
