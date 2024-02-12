# FAQs

## Shoudld I use the L1 norm or the L2 norm?
The Dice loss, originally proposed to utilize the L2 norm [7], is more frequently used with the L1 norm in the literature [3, 8, 9, 10, 11]. The two variants of the Jaccard and Dice loss are compared in [2, 6] and their results suggest a superior performance of the L1 norm over the L2 norm. Furthermore, prominent frameworks such as [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/997804c7510634dc8fd83f1194b434c60815a93e/nnunetv2/training/loss/dice.py#L8), [MONAI](https://github.com/Project-MONAI/MONAI/blob/ff430286c37e78d7592372a5a97377f0cbb0219c/monai/losses/dice.py#L30) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch/blob/6db76a1106426ac5b55f39fba68168f3bccae7f8/segmentation_models_pytorch/losses/dice.py#L12) all adopt the L1 norm as the default setting in their implementations.

## Should I combine the cross-entropy loss with JDTLoss?
In a survery study [12], among participants in 80 medical imaging competitions, 39% exclusively utilize the cross-entropy loss; 36% use a combination of the cross-entropy loss and the Dice loss; 26% only employ the Dice loss. Berman et al. [4] suggest an approach where models are initially trained with the cross-entropy loss and subsequently fine-tuned using the Lovasz-Softmax loss. Currently, the predominant method for training models on natural images involves integrating the cross-entropy loss and the Dice loss in equal proportions (1:1 ratio) [9, 10, 11]. However, we [1, 2, 3] find that, across natural, aerial, and medical image models, mixing the cross-entropy loss and JDTLoss with a ratio of 0.25:0.75 tends to yield superior results.

## How should I choose the training hyper-parameters?
Although training hyper-parameters (such as learning rate, weight decay, etc) are usually selected based on the cross-entropy loss, the adoption of JDTLoss does not require modifications to these hyperparameters [1, 2, 3]. However, it is important to highlight that the gradient of the cross-entropy loss plateaus as the prediction is close to the target, while the gradient of the Jaccard loss remains steeper [2]. This distinction implies that models trained solely with the cross-entropy loss may require longer epochs to converge. On the other hand, employing JDTLoss for extended epochs could potentially result in overfitting. In light of this, our experiments [1, 2, 3] normally reduce the number of epochs to half of the original value.

## Should I calculate the loss over each individual image or over all images in the mini-batch?
When computing the loss for a mini-batch, we can either calculate the loss for each individual image and then average these values, or compute a single loss value across all images within the mini-batch. In most existing [implementations](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/dice_loss.py) of the Jaccard, Dice and Tversky loss, the loss is aggregated over the entire mini-batch. However, the Lovasz-Softmax loss calculates the loss over all images ([`per_image = False`](https://github.com/bermanmaxim/LovaszSoftmax/blob/7d48792d35a04d3167de488dd00daabbccd8334b/pytorch/lovasz_losses.py#L53)) for multi-class segmentation, and averages over each individual image ([`per_image = True`](https://github.com/bermanmaxim/LovaszSoftmax/blob/7d48792d35a04d3167de488dd00daabbccd8334b/pytorch/lovasz_losses.py#L33)) for binary segmentation.

In general, we should align the loss function with the evaluation metric [1]. Therefore, if the evaluation metric is $\text{mIoU}^\text{D}$, aggregating the loss over the whole mini-batch is preferable. On the other hand, if the evaluation metric is $\text{mIoU}^\text{I}$ or $\text{mIoU}^\text{C}$, it is advisable to [reduce](https://github.com/zifuwanggg/JDTLosses/blob/e584fd80d9b4f7c672368517b2141cad02e8b8df/losses/jdt_loss.py#L209) the loss values accordingly.

In a distributed trainning setting with multiple GPUs, if the per-GPU batch size is small (which is often the case) and the loss is computed on each GPU independently, the loss function may inadvertently optimize for $\text{mIoU}^\text{I}$. This can lead to suboptimal outcomes when evaluated with $\text{mIoU}^\text{D}$.

## Which classes should I use to calcualte the loss value?
Loss functions used in semantic segmentation typically compute the loss for each class individually before averaging them. However, not every class appears in a mini-batch during training. Most existing [implementations](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/dice_loss.py) of the Jaccard, Dice and Tversky loss calculate the loss across all classes in the dataset, regardless of their presence in the current mini-batch. On the other hand, the Lovasz-Softmax only considers present classes ([`classes = "present"`](https://github.com/bermanmaxim/LovaszSoftmax/blob/7d48792d35a04d3167de488dd00daabbccd8334b/pytorch/lovasz_losses.py#L153)). For matching-based architectures such as Mask2Former [10], the Dice loss applied in the mask loss also only take present classes into account.

When using hard labels, we [2] and Berman et al. [4] suggest to optimize over present classes, since the loss aligns more effectively with the evaluation metric, especially when (i) the dataset contains a large number of classes (e.g. ADE20K), and/or (ii) evaluated with fine-grained evaluation metrics [1]. With soft labels, the optimal choice varies based on specific applications [2].

If there exists severe class imbalances, the class-wise loss may inadvertently amplify the significance of underrepresented classes, potentially disturb the training process. In such scenerio, one could consider adopting a class-agnostic loss computation, where the intersection and union are calculated over all pixels [13].

## Which loss should I use?
We should align the loss function with the evaluation metric. Therefore, when the evaluation metric is IoU, the Jaccard loss [2] is advised, while it is the Dice score, the Dice loss [3] is more appropriate. The Tversky loss [3] is useful when separate weights are required for false positives and false negatives. Furthermore, when dealing with fine-grained evaluation metrics, it is imperative to adjust the loss function accordingly [1].

It is also worth noting, as previously discussed, that while both serve as differentiable surrogates of the Jaccard index, the implementations of Jaccard loss and the Lovasz-Softmax loss [4, 5] often differ. However, when utilizing the same implementation (such as calculating the loss value over each individual image or over all images in the mini-batch; using which classes to calcualte the loss), their results tend to be remarkably similar [2, 6]. Additionally, it is important to recognize that the Lovasz-Softmax loss is not compatible with soft labels.

## References
[1] Zifu Wang, Maxim Berman, Amal Rannen-Triki, Philip H.S. Torr, Devis Tuia, Tinne Tuytelaars, Luc Van Gool, Jiaqian Yu, Matthew B. Blaschko. Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union. NeurIPS, 2023.

[2] Zifu Wang, Xuefei Ning, Matthew B. Blaschko. Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels. NeurIPS, 2023.

[3] Zifu Wang, Teodora Popordanoska, Jeroen Bertels, Robin Lemmens, Matthew B. Blaschko. Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels. MICCAI, 2023.

[4] Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko. The Lovasz-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks. CVPR, 2018.

[5] Jiaqian Yu, Matthew B. Blaschko. The Lovasz Hinge: A Novel Convex Surrogate for Submodular Losses. TPAMI, 2018.

[6] Tom Eelbode, Jeroen Bertels, Maxim Berman, Dirk Vandermeulen, Frederik Maes, Raf Bisschops, Matthew B. Blaschko. Optimization for Medical Image Segmentation: Theory and Practice When Evaluating With Dice Score or Jaccard Index. TMI, 2020.

[7] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV, 2016.

[8] Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 2021.

[9] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko. End-to-End Object Detection with Transformers. ECCV, 2020.

[10] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar. Masked-attention Mask Transformer for Universal Image Segmentation. CVPR, 2022.

[11] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick. Segment Anything. ICCV, 2023.

[12] Matthias Eisenmann et al. Why is the Winner the Best? CVPR, 2023.

[13] Carole H Sudre, Wenqi Li, Tom Vercauteren, Sébastien Ourselin, M. Jorge Cardoso. Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. MICCAI Workshop, 2017.