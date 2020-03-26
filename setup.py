from setuptools import setup, find_packages

setup(
    name='language_models', packages=find_packages(where='src'), package_dir={'': 'src'}
)
