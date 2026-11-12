#!/usr/bin/env bash
# Run on the server after sync_to_server.sh.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_server.sh

Environment overrides:
  DATASET_ROOT   default: ../data/r2s100k relative to project
  CHECKPOINT     default: checkpoints/sed_model_large.pth
  OUTPUT         default: runs/sed_l_train
  SMOKE_IMAGE    default: first image in train/ (set a path to choose one)
  WORKERS        default: 4
  MAX_JOBS       default: 2 for Detectron2 build

Creates config.server.yaml, installs .venv, then runs one-image inference
and evaluation. It does not start the full train-split run.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# != 0 )); then usage >&2; exit 2; fi
project_dir=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$project_dir"
dataset_root=${DATASET_ROOT:-$project_dir/../data/r2s100k}
checkpoint=${CHECKPOINT:-$project_dir/checkpoints/sed_model_large.pth}
output=${OUTPUT:-$project_dir/runs/sed_l_train}
workers=${WORKERS:-4}
dataset_root=$(realpath -m "$dataset_root")
checkpoint=$(realpath -m "$checkpoint")
output=$(realpath -m "$output")

command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver / nvidia-smi is unavailable.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v python3.10 >/dev/null || { echo 'Python 3.10 is required.' >&2; exit 1; }
command -v git >/dev/null || { echo 'git is required.' >&2; exit 1; }
command -v c++ >/dev/null || { echo 'A C++ compiler is required for Detectron2.' >&2; exit 1; }
[[ -s $checkpoint ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
[[ -d $dataset_root/train && -d $dataset_root/train_labels ]] || {
  echo "Missing train/ or train_labels/ under $dataset_root" >&2; exit 1;
}
[[ $workers =~ ^[0-9]+$ ]] || { echo 'WORKERS must be a nonnegative integer.' >&2; exit 2; }

export MAX_JOBS=${MAX_JOBS:-2}
echo '[setup] Installing isolated Python environment'
bash scripts/setup_env.sh

echo '[setup] Checking GPU, dependency versions and pinned SED source'
.venv/bin/python - <<'PY'
import torch
from sed_r2s100k.inference import verify_upstream
verify_upstream()
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name())
PY
.venv/bin/python -m pip check

export SED_SERVER_DATASET_ROOT="$dataset_root"
export SED_SERVER_CHECKPOINT="$checkpoint"
export SED_SERVER_OUTPUT="$output"
export SED_SERVER_WORKERS="$workers"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import yaml
config = yaml.safe_load(Path('config.yaml').read_text())
config['dataset_root'] = os.environ['SED_SERVER_DATASET_ROOT']
config['checkpoint'] = os.environ['SED_SERVER_CHECKPOINT']
config['output'] = os.environ['SED_SERVER_OUTPUT']
config['device'] = 'cuda'
config['workers'] = int(os.environ['SED_SERVER_WORKERS'])
Path('config.server.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
print('Wrote config.server.yaml')
PY

if [[ -n ${SMOKE_IMAGE:-} ]]; then
  smoke_image=$(realpath "$SMOKE_IMAGE")
else
  smoke_image=$(find "$dataset_root/train" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)
fi
[[ -f $smoke_image ]] || { echo "Smoke image not found: $smoke_image" >&2; exit 1; }
smoke_output=$project_dir/runs/server_smoke_$(date +%Y%m%d_%H%M%S)_$$
echo "[setup] Smoke image: $smoke_image"
.venv/bin/python -m sed_r2s100k --config config.server.yaml infer \
  --image "$smoke_image" --output "$smoke_output" --workers 0
.venv/bin/python -m sed_r2s100k --config config.server.yaml evaluate \
  --image "$smoke_image" --output "$smoke_output"

echo "[setup] Smoke test passed: $smoke_output"
echo '[setup] Next: .venv/bin/python -m sed_r2s100k --config config.server.yaml benchmark'
echo '[setup] Then: .venv/bin/python -m sed_r2s100k --config config.server.yaml run'
