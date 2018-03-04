from setuptools import setup

setup(
    name='gp3',
    packages = ['gp3', 'gp3.inference', 'gp3.likelihoods',
                'gp3.utils', 'gp3.kernels'],
    version='0.1.5',
    description='Gaussian Processes with Probabilistic Programming',
    author='Anuj Sharma',
    author_email="anuj.sharma@columbia.edu",
    install_requires=['numpy>=1.7',
                      'six>=1.10.0'],
    url='https://github.com/as4529/gp3',  # use the URL to the github repo
    extras_require={
        'notebooks': ['jupyter>=1.0.0'],
        'visualization': ['matplotlib>=1.3',
                          'plotly>=2.2.2',
                          'tqdm>=4.19.4'],
        'autograd': ['autograd>=1.2']})