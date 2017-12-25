**GP3**: **G**aussian **P**rocesses with **P**robabilistic **P**rogramming

**Overview**

gp3 currently focuses on grid structure-exploiting inference for Gaussian Process Latent Variable Models. As of now, we support Laplace approximation and Stochastic Variational Inference. For usage examples see the notebooks in the examples directory. Comprehensive documentation coming soon.

gp3 is currently compatible with Python 2.7. Install with

```pip install gp3```

**Custom likelihoods**

gp3 leverages [autograd](https://github.com/HIPS/autograd) to allow for custom likelihoods. Examples coming soon.

**References**

For more information on structure-exploiting inference for GPs, see the following:

Adams et al, *Tractable Nonparametric Bayesian Inference in Poisson Processes with Gaussian Process Intensities*, Proceedings of the 26th International Conference on Machine Learning, Montreal, Canada, 2009

Flaxman et al, *Fast Kronecker Inference in Gaussian Processes with non-Gaussian Likelihoods*, Proceedings of the 32nd$$ International Conference on Machine Learning, Lille, France, 2015

Wilson and Nickisch, *Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP)*, Proceedings of the 32nd International Conference on Machine Learning, Lille, France, 2015
