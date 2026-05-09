"""
Pytest configuration for local test imports.

This file ensures that the project root is available in sys.path,
allowing tests to import modules from src/ and scripts/.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))