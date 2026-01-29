## Outline
This project was developed as the final project for CPSC 440/540: Advanced Machine Learning at the University of British Columbia. 
The main contribution of this project is the KNVisionTransformer architecture, that utilizes KernelNorm from Nasirigerdeh et al. (2024) in place of other methods like LayerNorm or BatchNorm.
Kernel normalization takes into account spatial correlation between its inputs. As a result, it achieves lower training loss than other normalization methods given equivalent training times.
The model utilizes a custom transformer encoder block (EncoderBlockKN) to leverage kernel normalization and reshape input and output tensors as required.

## Full paper
The full paper for this project can be found in the root directory under `cpsc440-kernel-norm-paper.pdf`. I was responsible for the work on kernel-normalized vision transformers and my partner worked on DenseNets. We researched and wrote the final paper together, sharing progress along the way.
