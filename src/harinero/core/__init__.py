"""Core functionality for Harinero Tanda Creator.

This package provides the main interfaces for tanda creation and management,
including song finding, structure definitions, and tanda generation.

Key Components:
    - SongFinder: Database operations for finding and filtering songs
    - SongStruct: Data structure for song information
    - TandaStruct: Data structure for tanda organization
    - TandaGenerator: Creates and manages tandas
    - HarineroTandaCreator: Main interface for the tanda creation system
"""

from typing import List, Dict, Optional, Union, Any

# Import core components
from .database.finder import SongFinder
from .models.structures import SongStruct, TandaStruct
from .tanda.tanda_generator import TandaGenerator
from .tanda.harinero_tanda_creator import HarineroTandaCreator

# Type aliases for core operations
DatabaseQuery = Dict[str, Any]
SongMetadata = Dict[str, Union[str, int, float]]
TandaConfiguration = Dict[str, Any]

__all__ = [
    # Main components
    'SongFinder',
    'SongStruct',
    'TandaStruct',
    'TandaGenerator',
    'HarineroTandaCreator',

    # Type aliases
    'DatabaseQuery',
    'SongMetadata',
    'TandaConfiguration',
]