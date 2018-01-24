#!/usr/bin/env python
from setuptools import setup
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
      url='http://github.com/jzuhone/cluster_generator',
      download_url='https://github.com/jzuhone/cluster_generator/tarball/0.1.0',
      install_requires=["six","numpy","scipy","yt","cython"],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Visualization',
      ],
      ext_modules = cythonize([cython_utils]),
      )

