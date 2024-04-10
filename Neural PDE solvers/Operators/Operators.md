# Neural PDE solvers

Here one can find several examples how to train PDE solver with different architectures.

If GitHub fails to render notebook you can try [nbviewer](https://nbviewer.org/).

Currently, I implemented:
1. [DeepONet](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Operators/Deep%20Operator%20Network.ipynb): very basic version with Siren as a trunk network and MLP as a branch network.
2. [DilResNet](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Operators/Dilated%20Residual%20Network.ipynb): classical architecture with residual connections and blocks of dilated convolutions.
3. [FFNO](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Operators/Factorized%20Fourier%20Neural%20Operator.ipynb): factorized version of FNO.
4. [ViT](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Operators/Vision%20Transformer%20(PDE).ipynb): standard vision transformer ([this notebook](https://github.com/VLSF/afternotes/blob/main/Transformers/Vision%20Transformer.ipynb) also describes vision transformer)

To appear:
1. POD-DeepONet following https://arxiv.org/abs/2111.05512
2. Classical FNO, https://arxiv.org/abs/2010.08895
3. U-Net, https://arxiv.org/abs/1505.04597
