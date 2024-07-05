## Outline
This project was developed as the final project for CPSC 440/540: Advanced Machine Learning at the University of British Columbia. 
The main contribution of this project is the KNVisionTransformer architecture, that utilizes KernelNorm from Nasirigerdeh et al. (2024) in place of other methods like LayerNorm or BatchNorm.
Kernel normalization takes into account spatial correlation between its inputs. As a result, it achieves lower training loss than other normalization methods given equivalent training times.
The model utilizes a custom transformer encoder block (EncoderBlockKN) to leverage kernel normalization and reshape input and output tensors as required.
