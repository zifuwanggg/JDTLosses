# Optimization with JDTLoss and Evaluation with Fine-grained mIoUs for Semantic Segmentation

## Models
* Methods
  * [UNet](https://arxiv.org/abs/1505.04597)
  * [DeepLabV3](https://arxiv.org/abs/1706.05587)
  * [DeepLabV3+](https://arxiv.org/abs/1802.02611)
  * [PSPNet](https://arxiv.org/abs/1612.01105)
  * [UPerNet](https://arxiv.org/abs/1807.10221)
  * [SegFormer](https://arxiv.org/abs/2105.15203)
* Backbones
  * [ResNet18/34/50/101/152](https://arxiv.org/abs/1512.03385)
  * [ConvNeXt-B](https://arxiv.org/abs/2201.03545)
  * [Xception65](https://arxiv.org/abs/1610.02357)
  * [EfficientNet-B0](https://arxiv.org/abs/1905.11946)
  * [MobileNetV2](https://arxiv.org/abs/1801.04381)
  * [MiTB0/B1/B2/B3/B4](https://arxiv.org/abs/2105.15203)
  * [MobileViTV2](https://arxiv.org/abs/2206.02680)
  * [`timm`](https://github.com/huggingface/pytorch-image-models)

## Datasets
* Urban
  * [Cityscapes](https://arxiv.org/abs/1604.01685)
  * [Nighttime Driving](https://arxiv.org/abs/1810.02575)
  * [Dark Zurich](https://arxiv.org/abs/1901.05946)
  * [Mapillary Vistas](https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf)
  * [CamVid](https://link.springer.com/chapter/10.1007/978-3-540-88682-2_5)
* "Thing" & "stuff"
  * [ADE20K](https://arxiv.org/abs/1608.05442)
  * [COCO-Stuff](https://arxiv.org/abs/1612.03716)
  * [PASCAL VOC](https://link.springer.com/article/10.1007/s11263-009-0275-4)
  * [PASCAL Context](https://ieeexplore.ieee.org/document/6909514)
* Aerial
  * [DeepGlobe Land](https://arxiv.org/abs/1805.06561)
  * [DeepGlobe Road](https://arxiv.org/abs/1805.06561)
  * [DeepGlobe Building](https://arxiv.org/abs/1805.06561)
* Medical
  * [LiTS](https://arxiv.org/abs/1901.04056)
  * [KiTS](https://arxiv.org/abs/1904.00445)
  * [QUBIQ](https://qubiq.grand-challenge.org)

## Metrics
* Pixel-wise Accuracy
  * Acc, mAcc
* mIoU
  * $\text{mIoU}^\text{I,C,K,D}$
* Calibration Error
  * $\text{ECE}^\text{I,D}$
  * $\text{SCE}^\text{I,D}$

## Prerequisites
#### Requirements
* Software: `timm`
* Hardware: 1-4 NVIDIA P100/V100 or 1 NVIDIA A100, depending on the model/dataset/batch size/crop size

#### Data Preparation
Coming soon.

## Usage
* Use the loss in your codebase:
```
from losses.jdt_loss import JDTLoss

"""
The Jaccard loss (default): JDTLoss()
The Dice loss: JDTLoss(alpha=0.5, beta=0.5)
"""
criterion = JDTLoss()
logits = model(image)
loss = criterion(logits, label)
```

* Hard labels
```
python main.py \
  --output_dir "path/to/output_dir" \
  --data_dir "path/to/data_dir" \
  --model_yaml "deeplabv3plus_resnet101d" \
  --data_yaml "cityscapes" \
  --label_yaml "hard" \
  --loss_yaml "jaccard_ic_present_all" \
  --schedule_yaml "40k_iters" \
  --optim_yaml "adamw_lr6e-5" \
  --test_yaml "test_iou"
```

* Label smoothing...
* Knowledge distillation...
* Multiple annotators...

## FAQs
Please refer to [JML](https://arxiv.org/abs/2302.05666) for how to tune the hyper-parameter and how to use JDTLoss.

## Acknowledgements
We express our gratitude to the creators and maintainers of the following projects: [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch), [structure_knowledge_distillation](https://github.com/irfanICMLL/structure_knowledge_distillation)

## Citations
```BibTeX
@InProceedings{Wang2023Revisiting,
  title     = {Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union},
  author    = {Wang, Zifu and Berman, Maxim and Rannen-Triki, Amal and Torr, Philip H.S. and Tuia, Devis and Tuytelaars, Tinne and Van Gool, Luc and Yu, Jiaqian and Blaschko, Matthew B.},
  booktitle = {NeurIPS},
  year      = {2023}
}

@InProceedings{Wang2023Jaccard,
  title     = {Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels},
  author    = {Wang, Zifu and Ning, Xuefei and Blaschko, Matthew B.},
  booktitle = {NeurIPS},
  year      = {2023}
}

@InProceedings{Wang2023Dice,
  title     = {Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels},
  author    = {Wang, Zifu and Popordanoska, Teodora and Bertels, Jeroen and Lemmens, Robin and Blaschko, Matthew B.},
  booktitle = {MICCAI},
  year      = {2023}
}
```