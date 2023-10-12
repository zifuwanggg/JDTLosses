# Optimization with JDTLoss and evaluation with fine-grained mIoUs for semantic segmentation

Training scripts will be released by the end of October (at the latest).

## Benchmarks
Dataset | Model | SKD | IFVD | CWD | CIRKD | MasKD | DIST | JML-KD
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
Cityscapes | DL3-R18 | 75.42 | 75.59 | 75.55 | 76.38 | 77.00 | 77.10 | **78.14** 
Cityscapes | DL3-MB2 | 73.82 | 73.50 | 74.66 | 75.42 | 75.26 | - | **77.78**
Cityscapes | PSP-R18 | 73.29 | 73.71 | 74.36 | 74.73 | 75.34 | 76.31 | **77.75**
PASCAL VOC | DL3-R18 | 73.51 | 73.85 | 74.02 | 74.50 | - | - | **76.25**
PASCAL VOC | PSP-R18 | 74.07 | 73.54 | 73.99 | 74.78 | - | - | **75.36**

## Requirements
* Software: `timm`
* Hardware: 1-4 16GB GPUs, depending on the batch size and the crop size

## Datasets
* Cityscapes
* PASCAL VOC
* DeepGlobe Land

## Models
* Backbones
  * ResNet18/34/50/101/152
  * Xception65
  * EfficientNet
  * MobileNetV2
  * `timm` models
* Methods
  * UNet
  * PSPNet
  * DeepLabV3
  * DeepLabV3+

## Usage
* Coming soon.

## FAQs
### What is the difference between JMLs and the Lovasz-Softmax loss?
With hard labels, they usually obtain very similar results. However, the Lovasz-Softmax loss is incompatible with soft labels.

### What is the difference between JMLs and the soft Jaccard loss?
With hard labels, they are identical. However, the soft Jaccard loss is incompatible with soft labels. Practically, we find some features that are essential for training segmentation models are often missing/wrong in public implementations of the soft Jaccard loss, such as specifying active classes, class weights and ignored pixels. JMLs also include a focal term that can be helpful for highly imbalanced datasets. Therefore, we expect JMLs can outperform the soft Jaccard loss even if only hard labels are presented.

### Why I find JMLs perform worse than CE?
We notice that current training recipes are highly optimized for CE. Although we have shown in the paper that these training hyper-parameters still work for JMLs, the optimal hyper-parameters for JMLs, depending on your datasets and architectures, might be slightly different. For example, through our preliminary experiments, we find that models trained with JMLs usually converge much faster than CE. If a model is trained for excessively long epochs which is the case for many recent segmentation models, the performance with JMLs might degrade.

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