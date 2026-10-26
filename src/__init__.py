"""
spatial-he-vision-study4: InSituVision-AI Deep Attention Multiple Instance Learning
===================================================================================
Author: Luca Celebrano (School of Medicine, University of Naples Federico II & TIGEM)
License: GNU General Public License v3.0 (GPL-3.0)
DOI: 10.5281/zenodo.23621924
"""

__version__ = "1.0.0"
__author__ = "Luca Celebrano"

import os as _os
import sys as _sys

def _display_star_tip():
    if _os.environ.get("STUDY4_SILENT") or _os.environ.get("CI") or not _sys.stdout.isatty():
        return
    _sys.stdout.write(
        "\n\033[1;33m⭐ Enjoying InSituVision-AI?\033[0m Star the repo on GitHub: "
        "\033[4;34mhttps://github.com/Lukecele/spatial-he-vision-study4\033[0m\n\n"
    )
    _sys.stdout.flush()

try:
    _display_star_tip()
except Exception:
    pass
