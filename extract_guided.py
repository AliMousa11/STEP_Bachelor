#!/usr/bin/env python
# update_graph_with_knn_guidance.py - Apply KNN guidance to extracted graph

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Update extracted graph with KNN guidance")
    parser.add_argument("--input-dir", type=str, default="/content/drive/MyDrive/Bachelor/extracted graphs",
                       help="Directory containing extracted graph components")
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/Bachelor/guided graphs",
                       help="Directory to save updated graphs")
    parser.add_argument("--guidance-strength", type=float, default=0.5,
                       help="Strength of KNN guidance (0-1, higher means more KNN influence)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of optimization iterations")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate for optimization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run optimization on")
    return parser.parse_args()

def load_graph_components(input_dir):
    """Load extracted graph components."""
    print(f"Loading graph components from {input_dir}...")
    
    # Load components
    sampled_adj_path = os.path.join(input_dir, "sampled_adj.npz")
    knn_adj_path = os.path.join(input_dir, "knn_adj.npz")
    prob_adj_path = os.path.join(input_dir, "prob_adj.npz")
    
    sampled_adj = np.load(sampled_adj_path)["adj"]
    knn_adj = np.load(knn_adj_path)["adj"]
    
    try:
        prob_adj = np.load(prob_adj_path)["adj"]
    except:
        # If prob_adj doesn't have "adj" key, try the first array
        prob_adj = np.load(prob_adj_path)[list(np.load(prob_adj_path).keys())[0]]
    
    return sampled_adj, knn_adj, prob_adj

def update_graph_with_knn_guidance(sampled_adj, knn_adj, prob_adj, guidance_strength=0.5, 
                                  iterations=100, lr=0.01, device="cuda"):
    """
    Update the sampled adjacency matrix with guidance from KNN graph.
    
    Args:
        sampled_adj: Binary sampled adjacency matrix [N, N]
        knn_adj: KNN adjacency matrix [N, N]
        prob_adj: Edge probability matrix [N, N]
        guidance_strength: Strength of KNN guidance (0-1)
        iterations: Number of optimization iterations
        lr: Learning rate for optimization
        device: Device to run optimization on
        
    Returns:
        updated_adj: Updated binary adjacency matrix [N, N]
        edge_probs: Updated edge probabilities [N, N]
    """
    # Convert to PyTorch tensors
    sampled_adj_tensor = torch.tensor(sampled_adj, device=device).float()
    knn_adj_tensor = torch.tensor(knn_adj, device=device).float()
    prob_adj_tensor = torch.tensor(prob_adj, device=device).float()
    
    num_nodes = sampled_adj.shape[0]
    
    # Create a learnable edge probability matrix
    # Initialize with extracted probabilities
    edge_probs = nn.Parameter(torch.logit(prob_adj_tensor.clamp(0.01, 0.99)), requires_grad=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam([edge_probs], lr=lr)
    
    # Create loss functions
    bce_loss = nn.BCELoss()
    
    # Create mask to ignore self-loops
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
    
    print(f"Optimizing graph with KNN guidance (strength={guidance_strength})...")
    
    # Optimization loop
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Convert to probabilities
        curr_probs = torch.sigmoid(edge_probs)
        
        # KNN guidance loss (similar to STEP's step_loss function)
        knn_loss = bce_loss(curr_probs[mask], knn_adj_tensor[mask])
        
        # Conservation loss to maintain similar density as original
        orig_density = sampled_adj_tensor.mean()
        density_loss = (curr_probs.mean() - orig_density).abs()
        
        # Total loss
        loss = guidance_strength * knn_loss + (1 - guidance_strength) * density_loss
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f"  Iteration {i+1}/{iterations}, Loss: {loss.item():.4f}, "
                 f"KNN Loss: {knn_loss.item():.4f}, Density Loss: {density_loss.item():.4f}")
    
    # Get final probabilities
    final_probs = torch.sigmoid(edge_probs)
    
    # Binarize to create adjacency matrix
    # We use a threshold that preserves approximately the same number of edges
    if guidance_strength < 1.0:
        # Find threshold that maintains similar density
        orig_density = sampled_adj_tensor.mean()
        target_edges = int(orig_density * num_nodes * num_nodes)
        
        # Flatten and sort probabilities (excluding diagonal)
        flat_probs = final_probs[mask].detach().cpu().numpy()
        sorted_probs = np.sort(flat_probs)[::-1]  # Sort in descending order
        
        # Find threshold that gives target number of edges
        threshold = sorted_probs[min(target_edges, len(sorted_probs)-1)]
        print(f"Using threshold {threshold:.4f} to maintain {target_edges} edges")
    else:
        # Pure KNN guidance - use 0.5 threshold
        threshold = 0.5
        print(f"Using threshold {threshold} (pure KNN guidance)")
    
    # Apply threshold
    updated_adj = (final_probs > threshold).float()
    
    # Remove self-loops
    updated_adj = updated_adj * (~torch.eye(num_nodes, dtype=torch.bool, device=device)).float()
    
    return updated_adj.detach().cpu().numpy(), final_probs.detach().cpu().numpy()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load graph components
    sampled_adj, knn_adj, prob_adj = load_graph_components(args.input_dir)
    
    print(f"Original sampled graph: {sampled_adj.shape}, edges: {sampled_adj.sum()}")
    print(f"KNN graph: {knn_adj.shape}, edges: {knn_adj.sum()}")
    
    # Update graph with KNN guidance
    updated_adj, updated_probs = update_graph_with_knn_guidance(
        sampled_adj, knn_adj, prob_adj,
        guidance_strength=args.guidance_strength,
        iterations=args.iterations,
        lr=args.learning_rate,
        device=args.device
    )
    
    print(f"Updated graph: {updated_adj.shape}, edges: {updated_adj.sum()}")
    
    # Calculate similarity with KNN graph
    knn_overlap = np.logical_and(updated_adj > 0, knn_adj > 0).sum()
    knn_similarity = knn_overlap / (knn_adj.sum() + 1e-10)
    print(f"Similarity with KNN graph: {knn_similarity:.4f} "
         f"({knn_overlap} shared edges out of {knn_adj.sum()} KNN edges)")
    
    # Calculate difference from original
    changed_edges = np.abs(updated_adj - sampled_adj).sum() / 2  # Divide by 2 since adjacency is symmetric
    print(f"Changed {changed_edges:.0f} edges from original graph")
    
    # Save results
    np.savez_compressed(
        os.path.join(args.output_dir, "guided_graph.npz"),
        adj=updated_adj,
        probs=updated_probs,
        original_adj=sampled_adj,
        knn_adj=knn_adj
    )
    
    print(f"\nSaved guided graph to {args.output_dir}/guided_graph.npz")
    print(f"  Use with --guidance-strength={args.guidance_strength}")

if __name__ == "__main__":
    main()