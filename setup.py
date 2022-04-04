#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='DynamicQuery',
      version='1.0',
      description='experiments for claim matching pipeline',
      author='Michael Shliselberg',
      author_email='michael.shliselberg@uconn.edu',
      packages=setuptools.find_packages("src"),
      package_dir={"":"src"}
     )