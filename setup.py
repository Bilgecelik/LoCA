from setuptools import find_packages, setup

setup(
    name='loca',
    packages=find_packages(),
    version='0.1.0',
    description='Automated continual learning for online data',
    author='B. Celik',
    install_requires = ['ray',
                        'numpy',
                        'torch',
                        'avalanche',
                        'wandb'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
