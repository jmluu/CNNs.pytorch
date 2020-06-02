#!/usr/bin/env bash 

set -e 

CUDA_VISIBLE_DEVICES=3,4,5,6  python mxnet_yolo_v3.py \
    --network mobilenet1.0 \
    --dataset coco \
    --datapath /home/datadisk2/dataset/COCO/ \
    --gpus 0,1,2,3 --batch-size 32 -j 64 \
    --log-interval 100 \
    --lr-decay-epoch 220,250 \
    --epochs 280 \
    --warmup-epochs 2
