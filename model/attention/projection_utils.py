"""
Utility functions for saving and loading projection (nn.Module) objects using pickle.
Useful for debugging and reusing projection layers across different sessions.
"""
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


DEFAULT_SAVE_PATH = Path("./projection.pkl")


def save_projection(projection: nn.Module, filepath: Optional[str] = None):
    """
    Save a projection object to disk using pickle.
    
    Args:
        projection: nn.Module object to save (e.g., project_q, project_k, project_v)
        filepath: Optional path. If None, uses "./projection.pkl"
    
    Example (in debugger):
        >>> from model.attention.projection_utils import save_projection
        >>> save_projection(project_q, "./projections/project_q.pkl")
        # Saves to ./projections/project_q.pkl
    """
    if filepath is None:
        filepath = DEFAULT_SAVE_PATH
    else:
        filepath = Path(filepath)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(projection, f)
    
    print(f"✅ Projection saved to {filepath.absolute()}")


def load_projection(filepath: Optional[str] = None, device: str = "cuda") -> nn.Module:
    """
    Load a projection object from disk.
    
    Args:
        filepath: Optional path. If None, uses "./projection.pkl"
        device: Device to load the object to (default: "cuda")
    
    Returns:
        nn.Module object (e.g., project_q, project_k, project_v)
    
    Example:
        >>> from model.attention.projection_utils import load_projection
        >>> project_q = load_projection("./projections/project_q.pkl", device="cuda:0")
    """
    if filepath is None:
        filepath = DEFAULT_SAVE_PATH
    else:
        filepath = Path(filepath)
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Projection file not found: {filepath.absolute()}")
    
    with open(filepath, 'rb') as f:
        projection = pickle.load(f)
    
    # Move to device
    if isinstance(projection, nn.Module):
        projection = projection.to(device)
        projection.eval()  # Set to evaluation mode
    
    print(f"✅ Projection loaded from {filepath.absolute()} to {device}")
    return projection

