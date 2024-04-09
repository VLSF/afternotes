# Datasets

Notebooks with dataset generation:
1. [Burgers dataset](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Datasets/Burgers%20datasets.ipynb): pseudospectral solver, viscosity is mild (no pronounced shocks).
2. [Elliptic smooth](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Datasets/Elliptic_smooth.py): finite difference approximation, for that reason diffusion coefficient is smooth.
3. [KdV](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Datasets/KdV%20dataset.ipynb): propagation of a single soliton ([see wiki](https://en.wikipedia.org/wiki/Korteweg–De_Vries_equation#One-soliton_solution)), the task of NN is to simulate convection but with amplitude-dependent speed.

Generated datasets (not practical to generate them on the flight):
1. [Burgers equation](https://disk.yandex.ru/d/kkcRqJYIIkRUgw): the same as in [Burgers dataset](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Datasets/Burgers%20datasets.ipynb) above, here is [the script](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Datasets/Burgers.py) used for dataset generation.
2. [Smooth elliptic equation](https://disk.yandex.ru/d/fz9ghnFEDm90Qw): the same as in [Elliptic smooth](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Datasets/Elliptic_smooth.py) above, generated with [this script](https://github.com/VLSF/afternotes/blob/main/Neural%20PDE%20solvers/Datasets/Elliptic_smooth.py).
