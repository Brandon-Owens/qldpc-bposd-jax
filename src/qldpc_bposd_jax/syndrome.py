import numpy as np


def binary_syndrome(H: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Compute the syndrome of a binary error vector over F_2."""
    return (H @ error) % 2
