#!/bin/bash
# Install into the POLICY conda env (the one with torch + LiLa-WAM deps):
# XPolicyLab itself (for ModelTemplate / websocket server) plus the light extras
# this sub-project needs on top of the LiLa-WAM requirements.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}"
python -m pip install h5py omegaconf scipy opencv-python tqdm matplotlib
