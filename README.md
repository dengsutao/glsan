# GLSAN
GLSAN is a network for drone-view small object detection.
## Installation
Our source codes are mainly based on [Detectron2](https://github.com/facebookresearch/detectron2), see [Detectron2.installation](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
## Get Started
About the initialization of [Detectron2](https://github.com/facebookresearch/detectron2), please refer to [Detectron2.Getting_started](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md
).
### dataset transformation
To train the VisDrone and UAVDT dataset, you need transform them to coco format.
We provide './tools/txt2xml_\*.py' and './tools/xml2json_\*.py' to generate json files in coco format.
### dataset augmentation
The network in our paper is trained with the augmented datasets.
We provide './tools/crop_dataset.py' and './tools/sr_dataset.py' to conduct SARSA and LSRN to the original datasets.
### pretrained models
The pretrained models of our network can be downloaded at [Detectron2.model_zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).
You can directly download [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl) or [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl)
to '.torch/fvcore_cache/detectron2/ImageNetPretrained/MSRA/' of your 'home' directory.
Or they will be downloaded automatically when training.
### training
We provide "train_net.py" for network training.
To train a model with "train_net.py", first setup the corresponding datasets following [Detectron2.datasets](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
you need to put the transformed or augmented datasets into './datasets' directory.
The settings of VisDrone and UAVDT can be found in './glsan/data/datasets'.

To train with 8 GPUs, run:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --num-gpus 8
```


To train with 1 GPU, run:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2
```


### evaluation
To evaluate a model's performance, there are threee modes corresponding to three different
cropping strategies: NoCrop, UniformlyCrop, SelfAdaptiveCrop.
You can run following codes to switch the cropping strategy:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8 GLSAN.CROP UniformlyCrop
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8 GLSAN.CROP SelfAdaptiveCrop
```
To add super-resolution operation to the network, run:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8 GLSAN.CROP SelfAdaptiveCrop GLSAN.SR True
```

To acquire more parameters of our method, see './glsan/config/defaults.py' and './glsan/modeling/meta_arch/glsan.py'