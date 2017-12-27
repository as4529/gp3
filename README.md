# gp3

**G**aussian **P**rocesses with **P**robabilistic **P**rogramming

## Overview

gp3 currently focuses on grid structure-exploiting inference for Gaussian Process Regression. As of now, it supports Laplace approximation and Stochastic Variational Inference. For usage examples, see ```examples/basic.ipynb```. To view the notebooks with visualizations, I recommend using [nbviewer](https://nbviewer.jupyter.org/). Comprehensive documentation coming soon.

## Installation

gp3 is currently compatible with Python 2.7. Install with

```pip install gp3```

## Custom Likelihoods

gp3 leverages [autograd](https://github.com/HIPS/autograd) to allow for custom likelihoods. See ```examples/lif.py``` and ```examples/lif.ipynb``` for examples.

## References

For more information on structure-exploiting inference for GPs, see the following:

Flaxman, Seth, Wilson, Andrew Gordon, Neil, Daniel B., Nickish, Hannes, Smola, Alexander J. (2015). *Fast Kronecker Inference in Gaussian Processes with Non-Gaussian Likelihoods*. Proceedings of the 32nd International Conference on Machine Learning.

Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian processes for Machine Learning*.The MIT Press.

Wilson, Andrew Gordon, Gilboa, Elad, Nehorai, Arye, and Cunningham, John P. (2014). *Fast Kernel Learning for Multidimensional Pattern Extrapolation*. 27th Conference on Neural Information Processing Systems (NIPS 2014).

Wilson, Andrew Gordon, Dann, Christoph, Nickish Hannes (2015). *Thoughts on Massively Scalable Gaussian Processes*

Wilson, Andrew Gordon, Hu, Zhiting, Salakhutdinov, Ruslan, Xing, Eric P. (2016). *Stochastic Variational Deep Kernel Learning*. 29th Conference on Neural Information Processing Systems (NIPS 2016).
