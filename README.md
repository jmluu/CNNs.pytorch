# CNNs_PyTorch
 cnn training with bag of tricks

# Dependencies
- pytorch >= 1.0
- torchvision to load the datasets, perform image transforms
- tensorboardX 
- numpy 
- CUDA >= 9.0

# Experiment Results
## ImageNet 
**AlexNet**		acc1 = ??? 
```
CUDA_VISIBLE_DEVICES=1 python imagenet.py \
	--data /imagenet-dir \
	--arch alexnet \
	--lr 0.01 --lr-mode step --lr-decay-period 40 \
	--epoch 160 --batch-size 256  -j 8 \
	--weight-decay 0.00005 
```


**ResNet-18:**   acc1 = 71.02
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data /imagenet-dir \
	--arch resnet18 \
	--lr 0.2 --lr-mode cosine --epoch 120 --batch-size 512  -j 32 \
	--warmup-epochs 5  --weight-decay 0.0001 


```
**ResNet-50:**   acc1 = 77.75
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data /imagenet-dir \
	--arch resnet50 \
	--lr 0.2 --lr-mode cosine --epoch 120 --batch-size 512  -j 60 \
	--warmup-epochs 5  --weight-decay 0.0001 \
	--no-wd --label-smoothing --last-gamma

```

**MobileNet-V2-1.0:** acc1 = 71.93
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data /imagenet-dir \
	--arch mobilenet_v2 \
	--lr 0.05 --lr-mode cosine --epoch 150 --batch-size 256  -j 32 \
	--warmup-epochs 5  --weight-decay 0.00004 \
	--no-wd --label-smoothing 
```

## Cifar10 
**cifar_resnet20:** acc = 92.21
```
CUDA_VISIBLE_DEVICES=0  python cifar10.py \
	--arch cifar_resnet20 \
	--lr 0.1 --epoch 200 --batch-size 128  -j 2 \
	--lr-decay 0.1 --lr-decay-epoch 100,150 \
	--weight-decay 0.0001 
```

# Cite 
[Gluoncv model_zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html)
```
@inproceedings{he2019bag,
  title={Bag of tricks for image classification with convolutional neural networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2019}
}
```
