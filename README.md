# Fisher Information Loss

This repository contains code that can be used to reproduce the experimental
results presented in the paper:

Awni Hannun, Chuan Guo and Laurens van der Maaten. Measuring Data Leakage in
Machine-Learning Models with Fisher Information. 
[arXiv 2102.11673](https://arxiv.org/abs/2102.11673), 2021.

# Installation

The code requires Python 3.7+, [PyTorch
1.7.1+](https://pytorch.org/get-started/locally/), and torchvision 0.8.2+.

Create an Anaconda environment and install the dependencies:

```
conda create --name fil
conda activate fil
conda install -c pytorch pytorch torchvision
pip install gitpython 
```

# Usage

The script `fisher_information.py` computes the per-example FIL for the given
dataset and model. An example run is:

```
python fisher_information.py \
    --dataset mnist \
    --model least_squares
```

To see usage options for the script run:

```
python fisher_information.py --help
```

Other scripts in the repository are:
- `reweighted.py` : Run the iteratively reweighted Fisher information loss
  (IRFIL) algorithm.
- `model_inversion.py` : Attribute inversion experiments for a non-private
  model.
- `private_model_inversion.py` : Attribute inversion experiments for a private
  model.
- `test_jacobians.py` : Unit tests.

To run the full set of experiments in the accompanying paper:
```
cd scripts/ && ./run_experiments.sh
```

# Citing this Repository

If you use the code in this repository, please cite the following paper:

```
@article{hannun2021fil,
  title={Measuring Data Leakage in Machine-Learning Models with Fisher
    Information},
  author={Hannun, Awni and Guo, Chuan and van der Maaten, Laurens},
  journal={arXiv preprint arXiv:2102.11673},
  year={2021}
}
```

# License

This code is released under a CC-BY-NC 4.0 license. Please see the
[LICENSE](LICENSE) file for more information.

Please review Facebook Open Source [Terms of
Use](https://opensource.facebook.com/legal/terms) and [Privacy
Policy](https://opensource.facebook.com/legal/privacy).

