"""
Utility functions for saving and loading position_bias (RotaryEmbeddingESM) objects using pickle.
Useful for debugging and reusing position embeddings across different sessions.
"""
import pickle
from pathlib import Path
from typing import Optional
from .rope import RotaryEmbeddingESM


DEFAULT_SAVE_PATH = Path("./position_bias.pkl")


def save_position_bias(position_bias: RotaryEmbeddingESM, filepath: Optional[str] = None):
    """
    Save a position_bias object to disk using pickle.
    
    Args:
        position_bias: RotaryEmbeddingESM object to save
        filepath: Optional path. If None, uses "./position_bias.pkl"
    
    Example (in debugger):
        >>> from model.attention.position_bias_utils import save_position_bias
        >>> save_position_bias(position_bias)
        # Saves to ./position_bias.pkl
    """
    if filepath is None:
        filepath = DEFAULT_SAVE_PATH
    else:
        filepath = Path(filepath)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(position_bias, f)
    
    print(f"✅ Position bias saved to {filepath.absolute()}")


def load_position_bias(filepath: Optional[str] = None, device: str = "cuda") -> RotaryEmbeddingESM:
    """
    Load a position_bias object from disk.
    
    Args:
        filepath: Optional path. If None, uses "./position_bias.pkl"
        device: Device to load the object to (default: "cuda")
    
    Returns:
        RotaryEmbeddingESM object
    
    Example:
        >>> from model.attention.position_bias_utils import load_position_bias
        >>> position_bias = load_position_bias()
    """
    if filepath is None:
        filepath = DEFAULT_SAVE_PATH
    else:
        filepath = Path(filepath)
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Position bias file not found: {filepath.absolute()}")
    
    with open(filepath, 'rb') as f:
        position_bias = pickle.load(f)
    
    # Move to device
    if position_bias._cos_cached is not None:
        position_bias._cos_cached = position_bias._cos_cached.to(device)
    if position_bias._sin_cached is not None:
        position_bias._sin_cached = position_bias._sin_cached.to(device)
    if hasattr(position_bias, 'inv_freq'):
        position_bias.inv_freq = position_bias.inv_freq.to(device)
    
    print(f"✅ Position bias loaded from {filepath.absolute()}")
    return position_bias

