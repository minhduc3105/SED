#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -e third_party/SED/.git ]; then
  if [ -d .git ] && [ -f .gitmodules ]; then
    git submodule update --init --recursive third_party/SED
  else
    git clone https://github.com/xb534/SED.git third_party/SED
  fi
fi
git -C third_party/SED checkout db20d93c0ec8b0ce7fae267a10f826dfc7da15f5
python3.10 -m venv .venv
.venv/bin/python -m pip install -U pip wheel
.venv/bin/python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install 'numpy<2' 'Pillow==9.5.0' 'PyYAML>=6,<7' 'einops>=0.7,<1' 'ftfy>=6,<7' 'regex>=2023.0.0' 'timm==0.9.12' 'opencv-python-headless>=4.8,<5' 'pycocotools>=2.0.7,<3' 'scipy>=1.10,<2' 'imageio>=2,<3' 'protobuf==3.20.3'
.venv/bin/python -m pip install -e third_party/SED/open_clip
.venv/bin/python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6' --no-build-isolation
# Detectron2 pulls the newest TensorBoard, whose protobuf requirement conflicts
# with SED's OpenCLIP. TensorBoard 2.14.1 accepts protobuf 3.20.x.
.venv/bin/python -m pip install 'tensorboard==2.14.1'
# Detectron2 v0.6 uses the legacy PIL.Image.LINEAR alias; OpenCLIP pins protobuf 3.20.
.venv/bin/python -m pip install --no-deps 'Pillow==9.5.0' 'protobuf==3.20.3'
.venv/bin/python -m pip check
PYTHONPATH=third_party/SED .venv/bin/python - <<'PYCODE'
import torch, detectron2, open_clip, sed
from PIL import Image
import google.protobuf
assert hasattr(Image, 'LINEAR'), 'Detectron2 v0.6 requires Pillow<10'
assert google.protobuf.__version__.startswith('3.20.'), 'SED OpenCLIP requires protobuf 3.20.x'
print(torch.__version__, torch.cuda.is_available(), google.protobuf.__version__)
PYCODE
