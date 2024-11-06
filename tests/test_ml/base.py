from typing import Optional, Dict, Any, Tuple
import numpy as np
import pytest


class BaseMLTest:
    """Base class for ML-related tests

    Provides common setup and utility methods for ML component testing.
    """

    @pytest.fixture(autouse=True)
    def setup(self, test_root_dir, test_config):
        """Basic setup for ML tests"""
        self.config = test_config
        self.data_dir = test_root_dir / 'data'
        self._setup_test_data()

    def _setup_test_data(self) -> None:
        """Override this in specific test classes"""
        pass

    def assert_feature_shapes(
            self,
            features: Dict[str, np.ndarray],
            expected_shapes: Dict[str, Tuple[int, ...]]
    ) -> None:
        """Utility to verify feature shapes"""
        for name, expected in expected_shapes.items():
            assert name in features, f"Missing feature: {name}"
            assert features[name].shape == expected, \
                f"Wrong shape for {name}: expected {expected}, got {features[name].shape}"