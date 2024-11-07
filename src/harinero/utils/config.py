import json
from pathlib import Path
from typing import Dict, Union, Any


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from the specified path.

    Args:
        config_path: Path to the config file (can be string or Path object)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        JSONDecodeError: If config file is not valid JSON
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as file:
        config = json.load(file)

    return config