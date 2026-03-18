from setuptools import setup, find_packages

setup(
    name='pyscf_gvb',
    version='0.1.0',
    description='GVB (Generalized Valence Bond) solver for PySCF',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Rick Muller',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'pyscf',
    ],
)
