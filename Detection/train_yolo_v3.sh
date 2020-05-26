#!/usr/bin/env bash 

set -e 

CUDA_VISIBLE_DEVICES=6 python mxnet_yolo_v3.py \
    --network mobilenet1.0 \
    --dataset coco \
    --datapath /home/datadisk2/dataset/COCO/ \
    --gpus 0 --batch-size 8 -j 4 \
    --log-interval 100 \
    --lr-decay-epoch 5,8 \
    --epochs 10 \
    --warmup-epochs 2