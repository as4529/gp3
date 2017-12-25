from setuptools import setup

setup(
    name='gp3',
    packages = ['gp3', 'gp3.inference', 'gp3.likelihoods',
                'gp3.utils'],
    version='0.0.3',
    description='Gaussian Processes with Probabilistic Programming',
    author='Anuj Sharma',
    author_email="anuj.sharma@columbia.edu",
    install_requires=['numpy>=1.7',
                      'six>=1.10.0'],
    url='https://github.com/as4529/gp3',  # use the URL to the github repo
    extras_require={
        'modeling': ['tensorflow>=1.2.0rc0',
                     'GPy==1.8.4',
                     ''],
        'notebooks': ['jupyter>=1.0.0'],
        'visualization': ['matplotlib>=1.3',
                          'plotly>=2.2.2',
                          'tqdm>=4.19.4']})