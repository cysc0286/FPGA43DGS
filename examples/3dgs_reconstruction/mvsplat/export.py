"""Compatibility module/CLI for gaussian_generation.adapter.

Alias the implementation so existing imports and patch targets share one state.
"""
import sys
from gaussian_generation import adapter as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
