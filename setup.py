#!/usr/bin/env python
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define the Cython extension module
cython_utils = Extension(
    "cluster_generator.cython_utils",
    sources=["cluster_generator/cython_utils.pyx"],
    libraries=["m"],  # Standard math library for C
    include_dirs=[np.get_include()],
)

setup(
    name="cluster_generator",
    packages=["cluster_generator"],
    version="0.1.0",
    description="Generating equilibrium models of galaxy clusters.",
    author="John ZuHone",
    author_email="jzuhone@gmail.com",
    url="https://github.com/jzuhone/cluster_generator",
    download_url="https://github.com/jzuhone/cluster_generator/tarball/0.1.0",
    install_requires=[
        "numpy",
        "scipy>=1.11.4",
        "yt",
        "unyt",
        "cython",
        "ruamel.yaml",
        "h5py",
        "tqdm",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,
    ext_modules=cythonize([cython_utils]),
)
