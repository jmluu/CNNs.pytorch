# CNNs_PyTorch
 cnn training with bag of tricks

# Dependencies
- pytorch >= 1.0
- torchvision to load the datasets, perform image transforms
- tensorboardX 
- numpy 
- CUDA >= 9.0

# Experiment Results

**ResNet-18:**   acc1 = 71.02
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
	--data /imagenet-dir \
	--arch resnet18 \
	--lr 0.2 --lr-mode cosine --epoch 120 --batch-size 512  -j 32 \
	--warmup-epochs 5  --weight-decay 0.0001 


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
# Cite 

```
@inproceedings{he2019bag,
  title={Bag of tricks for image classification with convolutional neural networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2019}
}
[Gluoncv model_zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html)
```
