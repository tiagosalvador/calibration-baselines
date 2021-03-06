# Calibration Baselines

The goal of Calibration Baseliens is to provide a starting point into post-hoc calibration. It is challenging to compare post-hoc calibration methods as they require a pre-trained model whose choice greatly impacts the final calibration. Therefore all methods need to be compared based on the same pre-trained model. By implementing current state-of-the-art methods with simple and concise code, this repo will ease the burden of researchers starting in the field.

## Requirements

TBD

## Methods

Currently the following methods are included:
 - Temperature Scaling (TS)
 - Vector Scaling (VS)
 - Matrix Scaling (MS)
 - Matrix Scaling w/ ODIR (MS-ODIR)
 - Dirichlet w/ L2 regularization (Dir-L2)
 - Dirichlet w/ ODIR (Dir-ODIR)
 - Ensemble Temperature Scaling (ETS)
 - Accuracy preserving Isotonic Regression (IRM)
 - Accuracy preserving Isotonic Regression with Temperature Scaling (IRM-TS)
 - Isotonic Regression One vs All (IROvA)
 - Isotonic Regression One vs All with Temperature Scaling (IROvA-TS)

## Getting started

Create a folder named 'datasets' and place your datasets there.

## Methods

Currently the following datasets are supported:
 - [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [SVHN](http://ufldl.stanford.edu/housenumbers)
 - [ImageNet-1K](http://www.image-net.org)

The following datasets are available to evaluate ood calibration:
 -  [CIFAR-10/100-C](https://github.com/hendrycks/robustness)
 -  [STL10](https://cs.stanford.edu/~acoates/stl10/)
 -  [CIFAR-10.1 (v4, v6)](https://github.com/modestyachts/CIFAR-10.1)

To download all datasets, except ImageNet, simply run
```bash
python download_datasets.py
```


## Training/Evaluation

To train and evaluate the post-hoc calibration methods:

```bash
python run_all_experiments.py
```

## Results

The results will be saved in the folder `data`. 

## References

Methods from the following papers were added:
 - [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
 - [Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration](https://arxiv.org/abs/1910.12656)
 - [Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning](https://arxiv.org/abs/2003.07329)

To be added:
 - [Intra Order-preserving Functions for Calibration of Multi-Class Neural Networks](https://arxiv.org/abs/2003.06820)

## Acknowledgement

Given that the focus is on post-hoc calibration methods, we use pre-trained models, which are obtained from [imgclsmob](https://github.com/osmr/imgclsmob), (already cloned into this repo).

The code for Matrix Scaling, Diriclet Calibration and respective variants was adapted from official code repository for Dirichlet Calbiration: [Dirichlet Calibration Python implementation](https://github.com/dirichletcal/dirichlet_python).
