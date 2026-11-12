#!/usr/bin/env bash
# Run locally: sync the SED project, checkpoint and R2S100K train split.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/sync_to_server.sh USER@HOST /absolute/server/workdir [local_dataset_root]

Remote layout:
  WORKDIR/SED/                  project and checkpoint
  WORKDIR/data/r2s100k/        train/ and train_labels/

Environment: SSH_PORT=22, RSYNC_DRY_RUN=1 (preview only).
The remote account needs SSH access and enough free space for about 8 GB.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# < 2 || $# > 3 )); then usage >&2; exit 2; fi
remote=$1
remote_root=${2%/}
local_root=${3:-/run/media/minhduc/D/Study/r2s100k}
port=${SSH_PORT:-22}
project_dir=$(cd "$(dirname "$0")/.." && pwd -P)

# Remote arguments are used in an SSH command; restrict them to shell-safe paths.
if [[ ! $remote =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]]; then
  echo 'Expected USER@HOST (letters, digits, dots, underscores, hyphens).' >&2; exit 2
fi
if [[ ! $remote_root =~ ^/[A-Za-z0-9._/-]+$ || $remote_root == / ]]; then
  echo 'Remote workdir must be an absolute path without spaces or shell characters.' >&2; exit 2
fi
if [[ ! $port =~ ^[0-9]+$ ]]; then echo 'SSH_PORT must be numeric.' >&2; exit 2; fi
for directory in train train_labels; do
  [[ -d $local_root/$directory ]] || { echo "Missing $local_root/$directory" >&2; exit 1; }
done
checkpoint=$project_dir/checkpoints/sed_model_large.pth
[[ -s $checkpoint ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
[[ -e $project_dir/third_party/SED/.git ]] || { echo 'Missing pinned SED source; run scripts/setup_env.sh first.' >&2; exit 1; }
command -v rsync >/dev/null || { echo 'rsync is required locally.' >&2; exit 1; }
command -v ssh >/dev/null || { echo 'ssh is required locally.' >&2; exit 1; }

rsync_flags=(-a --partial --info=progress2)
if [[ ${RSYNC_DRY_RUN:-0} == 1 ]]; then rsync_flags+=(--dry-run); fi
remote_project=$remote_root/SED
remote_data=$remote_root/data/r2s100k
ssh -p "$port" "$remote" "mkdir -p '$remote_project/checkpoints' '$remote_data/train' '$remote_data/train_labels'"

echo '[sync] Project source (excluding local environment and old outputs)'
rsync "${rsync_flags[@]}" -e "ssh -p $port" \
  --exclude='/.venv/' --exclude='/runs/' --exclude='/checkpoints/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  "$project_dir/" "$remote:$remote_project/"

echo '[sync] SED-L checkpoint'
rsync "${rsync_flags[@]}" -e "ssh -p $port" \
  "$checkpoint" "$remote:$remote_project/checkpoints/sed_model_large.pth"

for directory in train train_labels; do
  echo "[sync] R2S100K $directory"
  rsync "${rsync_flags[@]}" -e "ssh -p $port" \
    "$local_root/$directory/" "$remote:$remote_data/$directory/"
done

if [[ ${RSYNC_DRY_RUN:-0} == 1 ]]; then
  echo '[sync] Preview complete; no files were transferred.'
else
  local_hash=$(sha256sum "$checkpoint" | cut -d' ' -f1)
  remote_hash=$(ssh -p "$port" "$remote" "sha256sum '$remote_project/checkpoints/sed_model_large.pth'" | cut -d' ' -f1)
  [[ $local_hash == "$remote_hash" ]] || { echo 'Checkpoint checksum mismatch after transfer.' >&2; exit 1; }
  echo "[sync] Complete. Checkpoint SHA256: $local_hash"
  echo "[sync] On server: cd '$remote_project' && bash scripts/setup_server.sh"
fi
