### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="sc_filter",
    version="0.1.0",
    author="Open Source Bio-Computational Pipeline Contributors",
    description="A lightweight scRNA-seq denoising library for accessible genomic analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "torch>=1.11.0",
    ],
    python_requires=">=3.8",
)
