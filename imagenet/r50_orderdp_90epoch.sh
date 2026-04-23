#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 prune_experiment_orderdp.py -a resnet50 --dist-url 'tcp://127.0.0.1:12346' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ~/imagenet/data -b 1024 --lr 6.4 --epochs 90 > r50_0.75_0.8_orderdp_log_90epoch.txt
python3 prune_experiment_orderdp.py -a resnet18 --dist-url 'tcp://127.0.0.1:12346' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ~/imagenet/data -b 512 --lr 1.98 --random_len_ratio 0.8 --top_q_ratio 0.9 --epochs 90 > r18_0.8_0.9_orderdp_log_90epoch.txt
