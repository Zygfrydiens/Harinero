"""Core tanda generation functionality for the Harinero project.

This package provides tools for creating and managing tandas (sets of tango songs)
and generating milonga (dance event) structures.

Components:
    - TandaGenerator: Creates and manages tandas based on song similarity
    - HarineroTandaCreator: High-level interface for tanda creation
"""

from typing import List, Dict, Optional, Union

from .tanda_generator import TandaGenerator
from .harinero_tanda_creator import HarineroTandaCreator

# Type aliases for tanda-specific operations
TandaConfig = Dict[str, Union[int, float, str]]
SongList = List[Dict[str, Union[int, str, float]]]
TandaMetrics = Dict[str, float]

__all__ = [
    # Main components
    'TandaGenerator',
    'HarineroTandaCreator',

    # Type aliases
    'TandaConfig',
    'SongList',
    'TandaMetrics',
]