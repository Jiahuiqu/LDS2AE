# LDS $^2$ AE
The python code implementation of the AAAI 2024 paper "LDS$^2$AE: Local Diffusion Shared-Specific Autoencoder for Multimodal Remote Sensing Image Classification with Arbitrary Missing Modalities"


# Requirements

- Ubuntu 20.04   cuda 11.0
- Python 3.7  Pytorch 1.7

# Usage
We have presented test cases of the proposed model in config.py file.
- More details are commented in the code.

## Hyperparameters

The windowsize is set to 11 for Trento and Berlin，and is set to 27 for Houston2013！

The train_num_perclass is set to 40.

The optimizer is Adam.

The more detailed training settings are shown in experiments of this paper.

Training & Testing
just run the LDS2AE_main.py

# Cite
If you find this code helpful, please kindly cite:

```
@article{Qu_Yang_Dong_Yang_2024, 
title={LDS2AE: Local Diffusion Shared-Specific Autoencoder for Multimodal Remote Sensing Image Classification with Arbitrary Missing Modalities}, 
volume={38}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/29391}, 
DOI={10.1609/aaai.v38i13.29391}, 
number={13}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Qu, Jiahui and Yang, Yuanbo and Dong, Wenqian and Yang, Yufei}, 
year={2024}, 
month={Mar.}, 
pages={14731-14739} }

```

