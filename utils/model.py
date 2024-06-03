"""
    MIT License
    Copyright (c) 2024 Reza NasiriGerdeh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

# # import ResNet models
# from models.resnet.resnet_nn import resnet18_nn, resnet34_nn, resnet50_nn
# from models.resnet.resnet_bn import resnet18_bn, resnet34_bn, resnet50_bn
# from models.resnet.resnet_ln import resnet18_ln, resnet34_ln, resnet50_ln
# from models.resnet.resnet_gn import resnet18_gn, resnet34_gn, resnet50_gn

# # import PreactResNet models
# from models.preact_resnet.preact_resnet_nn import preact_resnet18_nn, preact_resnet34_nn, preact_resnet50_nn
# from models.preact_resnet.preact_resnet_bn import preact_resnet18_bn, preact_resnet34_bn, preact_resnet50_bn
# from models.preact_resnet.preact_resnet_ln import preact_resnet18_ln, preact_resnet34_ln, preact_resnet50_ln
# from models.preact_resnet.preact_resnet_gn import preact_resnet18_gn, preact_resnet34_gn, preact_resnet50_gn

# import KNResNet models
from models.knresnet import knresnet18, knresnet34, knresnet50

# import KNConvNeXt models
from models.knconvnext import knconvnext_tiny, knconvnext_small, knconvnext_base, knconvnext_large

import torch
from models.toynet import CNN
from models.vision_transformer import vit_b_16_ln, vit_s_16, vit_s_8, vit_b_16_bn
from models.knvision_transformer import knvit_s_16, knvit_s_8


import logging
logger = logging.getLogger("model")


# ADD MODELS AS WE CONSTRUCT MORE
def build_model(model_config: dict) -> torch.nn.Module:

    model_name = model_config['model_name']
    num_classes = model_config['num_classes']
    num_groups = model_config['num_groups']
    low_resolution = model_config['low_resolution']
    kn_dropout_p = model_config['kn_dropout_p']

    logger.info(f"Building model {model_name} ...")

    # #### Simple convolutional model
    if model_name == 'cnn':
        model = CNN(num_classes=num_classes)

    # ##### KN ViT
    elif model_name == 'knvit_s_16':
        model = knvit_s_16(num_classes=num_classes)
    
    elif model_name == 'knvit_s_8':
        model = knvit_s_8(num_classes=num_classes)

    # ##### Vision Transformers
    elif model_name == 'vit_s_16':
        model = vit_s_16(num_classes=num_classes)
    
    elif model_name == 'vit_s_8':
        model = vit_s_8(num_classes=num_classes)

    elif model_name == 'vit_b_16_ln':
        model = vit_b_16_ln(num_classes=num_classes)

    elif model_name == 'vit_b_16_bn':
        model = vit_b_16_bn(num_classes=num_classes)

    else:
        logger.error(f'{model_name} is an valid model name!')
        logger.info(f'You can add the model definition to build_model in utils/model.py')
        logger.info("Exiting ....")
        exit()

    logger.info(model)
    print()

    return model