"""Image decoding for RMBench HDF5 frames.

ALWAYS decode through ``decode_image_bit``: it returns RGB for both the legacy
channel-reversed JPEGs and the marked standard JPEGs. Never add a BGR<->RGB swap
after it (see the docstring of the vendored file).

Prefers the installed XPolicyLab implementation; falls back to the vendored copy
(rmbench@7d1cdfc data/decode_image_bit.py, MIT license) so that training does not
require XPolicyLab to be installed.
"""

from __future__ import annotations

try:  # pragma: no cover - depends on environment
    from XPolicyLab.utils.process_data import decode_image_bit, _has_rgb_marker  # type: ignore
    CODEC_SOURCE = "XPolicyLab.utils.process_data"
except Exception:  # ImportError, or XPolicyLab import-time failures
    from ._vendor.xpl_image_codec import decode_image_bit, _has_rgb_marker
    CODEC_SOURCE = "rmbench_lila._vendor.xpl_image_codec"

__all__ = ["decode_image_bit", "has_rgb_marker", "CODEC_SOURCE"]


def has_rgb_marker(image_bit) -> bool:
    """True for the standard (marked) format, False for the legacy format."""
    return bool(_has_rgb_marker(image_bit))
