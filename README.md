# Optimization with JDTLoss and Evaluation with Fine-grained mIoUs for Semantic Segmentation

Training scripts will be released by the end of October (at the latest).

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

* Hard labels...
* Label smoothing...
* Knowledge distillation...
* Multiple annotators...

## FAQs
### What is the difference between JMLs and the Lovasz-Softmax loss?
With hard labels, they usually obtain very similar results. However, the Lovasz-Softmax loss is incompatible with soft labels.

### What is the difference between JMLs and the soft Jaccard loss?
With hard labels, they are identical. However, the soft Jaccard loss is incompatible with soft labels. Practically, we find some features that are essential for training segmentation models are often missing/wrong in public implementations of the soft Jaccard loss, such as specifying active classes, class weights and ignored pixels. JMLs also include a focal term that can be helpful for highly imbalanced datasets. Therefore, we expect JMLs can outperform the soft Jaccard loss even if only hard labels are presented.

### Why I find JMLs perform worse than CE?
We notice that current training recipes are highly optimized for CE. Although we have shown in the paper that these training hyper-parameters still work for JMLs, the optimal hyper-parameters for JMLs, depending on your datasets and architectures, might be slightly different. For example, through our preliminary experiments, we find that models trained with JMLs usually converge much faster than CE. If a model is trained for excessively long epochs which is the case for many recent segmentation models, the performance with JMLs might degrade.

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

@InProceedings{Wang2023JML,
  title     = {Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels},
  author    = {Wang, Zifu and Ning, Xuefei and Blaschko, Matthew B.},
  booktitle = {NeurIPS},
  year      = {2023}
}

@InProceedings{Wang2023DML,
  title     = {Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels},
  author    = {Wang, Zifu and Popordanoska, Teodora and Bertels, Jeroen and Lemmens, Robin and Blaschko, Matthew B.},
  booktitle = {MICCAI},
  year      = {2023}
}
```