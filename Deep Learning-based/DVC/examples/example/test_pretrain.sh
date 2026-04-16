#!/bin/bash
set -e

ROOT=savecode
export PYTHONPATH=$PYTHONPATH:$(pwd)/$ROOT

CUDA_VISIBLE_DEVICES=0 python -u $ROOT/main.py \
  --log log_scvcd_pretrain512_crf26.txt \
  --testscvcd7 \
  --scvcd_seq_root /data/lichaofei/data/SCVCD-NEW/test/sequences \
  --scvcd_rec_root /data/lichaofei/data/SCVCD-NEW/test/DVC_rec_I \
  --scvcd_list /data/lichaofei/data/SCVCD-NEW/test/DVC_test \
  --scvcd_refdir H265L26 \
  --pretrain /home/chaofeili/DVC/examples/example/snapshot/512.model \
  --config /home/chaofeili/DVC/examples/example/config_512.json