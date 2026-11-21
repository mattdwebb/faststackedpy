"""Native corrective-weighted stacked DiD; no cross-language runtime."""
__version__ = '0.1.0'
from .core import fit, StackedError

__all__ = ["fit", "StackedError"]
