from typing import Dict, Any, Optional
import numpy as np


def create_test_features() -> Dict[str, np.ndarray]:
    """Creates sample feature data for testing"""
    return {
        "mfcc": np.random.rand(20, 13),
        "spectral": np.random.rand(20, 7)
    }

__all__ = ["create_test_features"]
