# notes
- KernelNorm paper: https://arxiv.org/abs/2205.10089
    - https://github.com/reza-nasirigerdeh/norm-torch
- earlier privacy-preserving KernelNorm paper https://arxiv.org/abs/2210.00053
    - https://github.com/reza-nasirigerdeh/dp-torch
    - https://github.com/reza-nasirigerdeh/fl-torch
    - has densenet KernelNorm implementation and talks about performance of it, but for differentially private and/or federated learning only
- ResNet paper: https://arxiv.org/abs/1512.03385 

## proposal ideas
potential architectures to try with kernelnorm:
- DenseNet: https://arxiv.org/abs/1608.06993
    - each layer is is a concatenation of every other previous layer (in ResNets it is a sum)
- ConvNext: https://arxiv.org/abs/2201.03545
    - depth wise convolutions + ResNets
    - kernelnorm authors briefly touched on it but they didn't spend much much time on it: "Note that due to the time constraint and the limited resources we had as an academic institution, we could not spend much time on optimizing the ConvNext architecture for KernelNorm. That is, the accuracy gain from KernelNorm compared to LayerNorm for the ConvNext models could further be improved by tuning the architecture for KernelNorm." (from https://openreview.net/forum?id=Uv3XVAEgG6)
- Vision Transformer (ViT): https://arxiv.org/abs/2010.11929
    - transformer for images
    - replace layernorms with kernelnorms
    - "It is worth mentioning that LayerNorm can be considered as a special case of KernelNorm, where the kernel size is (width, height) of the input tensor and dropout probability is zero. Thus, KernelNorm is indeed applicable to any architecture whose underlying normalization layer is LayerNorm such as vision transformers." (from https://openreview.net/forum?id=Uv3XVAEgG6)
- Conformer: https://arxiv.org/abs/2105.03889
    - combines transformer (ViT) and cnn (ResNet)
    - many different ways to apply kernelnorm, could experiment with adding kernelnorm in different places, especially in place of layernorm because of above
- ConvMixer: https://arxiv.org/abs/2201.09792
    - 


datasets:
- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
    - image classification, 50k train 10k test, 32x32, 100 classes
    - preprocessing: "We adopt the data preprocessing and augmentation scheme widely used for the dataset (Huang et al., 2017a; He et al., 2016b;a): Horizontally flipping and randomly cropping the samples after padding them. The cropping and padding sizes are 32×32 and 4×4, respectively." (from kernelnorm paper)
- cityscapes: https://www.cityscapes-dataset.com/
    - image segmentation, 2975 train 500 validation, 2048x1024 (but cropped to 1024x512 in preprocessing), 30 classes
    - preprocessing: "Following Sun et al. (2019); Ortiz et al. (2020), the train samples are randomly cropped from 2048×1024 to 1024×512, horizontally flipped, and randomly scaled in the range of [0.5, 2.0]. The models are tested on the validation images, which are of shape 2048×1024" (from kernelnorm paper)

start with CIFAR-100 dataset and smallest version of each model, then if time permits try bigger models and/or cityscapes?