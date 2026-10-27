#!/bin/bash
# Install into the POLICY conda env (LiLaWAM: torch + transformers + LiLa-WAM deps).
#
# XPolicyLab is installed WITHOUT its dependency list, so pip cannot replace packages the
# LiLa-WAM environment already pins (opencv, numpy, ...). Only the websocket/serialization
# packages the policy server needs are added, and pip is asked to keep everything else.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install --no-deps -e "${XPL_ROOT}"
python -m pip install --upgrade-strategy only-if-needed "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5"
python - <<'PY'
import importlib
for m in ["torch", "transformers", "huggingface_hub", "websockets", "msgpack", "msgpack_numpy", "pydantic", "cv2", "h5py", "omegaconf", "scipy"]:
    mod = importlib.import_module(m)
    print(f"{m:16s} {getattr(mod, '__version__', 'ok')}")
from XPolicyLab.model_template import ModelTemplate  # noqa: F401
print("XPolicyLab importable: OK")
PY
