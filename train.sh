#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING='/home/Eric-chuan/workspace/dataset/pms_triplet/sequences/00001'

python train_pms.py --dataset pms_triplet --batch_size 12 --epochs 16 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--lr 0.003 \
--trainpath=$MVS_TRAINING \
--logdir ./checkpoints $@
