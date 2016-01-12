#!/usr/bin/env python
from setuptools import setup

setup(name='equilibrate',
      packages=['equilibrate'],
      version='0.1.0',
      description='Generating gravitational equilbrium models',
      author='John ZuHone',
      author_email='jzuhone@gmail.com',
      url='http://github.com/jzuhone/equilibrate',
      download_url='https://github.com/jzuhone/equilibrate/tarball/0.1.0',
      install_requires=["six","numpy","scipy","yt"],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Visualization',
      ],
      )
