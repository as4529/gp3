# gp3

Gaussian Processes with Probabilistic Programming

## Overview

gp3 currently focuses on grid structure-exploiting inference for Gaussian Process Regression with custom likelihoods and kernels. As of now, it supports inference via Laplace approximation and Stochastic Variational Inference. For an overview of the inference methods, see ```methods_overview.pdf```

For a basic example, see ```examples/basic.ipynb```. For an example of a custom likelihood see ```examples/lif.ipynb```. To view the notebooks with visualizations, use [nbviewer](https://nbviewer.jupyter.org/). Comprehensive documentation coming soon.

Compatible with Python 3. Install with 

```pip3 install gp3```

## Features

There are already a couple of nice libraries for GP inference in Python: [GPy](https://github.com/SheffieldML/GPy) and [GPFlow](https://github.com/GPflow/GPflow), as well as one in Matlab, [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/). Each of these libraries focuses on a different aspect of accessible GP inference. gp3's focuses are the following:

**Structure Exploiting Inference**

gp3 exclusively implements Gaussian Process inference that exploits grid structure in covariates (X). This currently includes methods that leverage Kronecker and Toeplitz structure, and will soon include inducing point methods that can leverage grid structure without requiring it in the data itself. See the references at the bottom for background on these approaches.

**Custom Likelihoods and Kernels**

gp3 currently leverages [autograd](https://github.com/HIPS/autograd) to allow for inference on custom likelihoods and kernels. I am working on transitioning to PyTorch to allow for easier integration with existing deep learning models.  See ```examples/lif.py``` and ```examples/lif.ipynb``` for examples of a custom likelihood function.

## Roadmap

**In Progress:**

* Accurate posterior variance estimates
* Inference for Multi-output GPs

**Next:**

* Inducing Points
* Deep Kernel Learning

## References

For more information on structure-exploiting inference for GPs, see the following:

Flaxman, Seth, Wilson, Andrew Gordon, Neil, Daniel B., Nickish, Hannes, Smola, Alexander J. (2015). *Fast Kronecker Inference in Gaussian Processes with Non-Gaussian Likelihoods*. Proceedings of the 32nd International Conference on Machine Learning.

Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian processes for Machine Learning*.The MIT Press.

Wilson, Andrew Gordon, Gilboa, Elad, Nehorai, Arye, and Cunningham, John P. (2014). *Fast Kernel Learning for Multidimensional Pattern Extrapolation*. 27th Conference on Neural Information Processing Systems (NIPS 2014).

Wilson, Andrew Gordon, Dann, Christoph, Nickish Hannes (2015). *Thoughts on Massively Scalable Gaussian Processes*

Wilson, Andrew Gordon, Hu, Zhiting, Salakhutdinov, Ruslan, Xing, Eric P. (2016). *Stochastic Variational Deep Kernel Learning*. 29th Conference on Neural Information Processing Systems (NIPS 2016).
