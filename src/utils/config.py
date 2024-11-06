import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def load_config(config_name: str = 'config.json') -> Dict[str, Any]:
    """Load configuration from the configs directory.

    Args:
        config_name: Name of the config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path('configs') / config_name
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config