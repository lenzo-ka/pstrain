"""Public API for packaging a trained model for distribution.

Packaging is the last step of the documented workflow, so it belongs on the
supported surface rather than behind ``pstrain.lib.steps.package``.
"""

from pstrain.lib.steps.package import create_noisedict, package_model

__all__ = [
    "create_noisedict",
    "package_model",
]
