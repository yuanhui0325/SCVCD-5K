#ROOT=savecode/
#export PYTHONPATH=$PYTHONPATH:$ROOT
#mkdir snapshot
#CUDA_VISIBLE_DEVICES=0,1 python -u $ROOT/main.py --log log.txt --config config.json
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT

export CUDA_VISIBLE_DEVICES=3,4

# 每个lambda单独输出目录
export SNAPSHOT_DIR="$(pwd)/snapshot256"
export EVENT_DIR="$(pwd)/events256"
mkdir -p "$SNAPSHOT_DIR" "$EVENT_DIR"

python -u $ROOT/main.py \
  --log log256.txt \
  --config config_256.json \
  --pretrain snapshot/256.model
