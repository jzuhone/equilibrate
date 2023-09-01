#!/usr/bin/env python
import os

from setuptools import setup

# - Force system to install cython (called before dependancy assertion) -#
os.system("pip install cython")

from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

cython_utils = Extension("cluster_generator.cython_utils",
                         sources=["cluster_generator/cython_utils.pyx"],
                         language='c', libraries=["m"],
                         include_dirs=[np.get_include()])

setup(name='cluster_generator',
      packages=['cluster_generator'],
      version='0.1.0',
      description='Generating equilbrium models of galaxy clusters.',
      author='John ZuHone',
      author_email='jzuhone@gmail.com',
      url='https://github.com/jzuhone/cluster_generator',
      download_url='https://github.com/jzuhone/cluster_generator/tarball/0.1.0',
      install_requires=["numpy", "scipy", "yt", "unyt", "cython",
                        "ruamel.yaml","tqdm","dill","halo","pandas"],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Visualization',
      ],
      package_data={
          "cluster_generator.bin..collections":["*.csv","*.yaml"],
          "cluster_generator.bin..resources":["*.yaml"]
      },
      ext_modules=cythonize([cython_utils]),
      )
