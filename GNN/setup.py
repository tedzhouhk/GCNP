# cython: language_level=3
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
# import cython_utils

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(ext_modules = cythonize(["GNN/cython_sampler.pyx","GNN/cython_utils.pyx","GNN/norm_aggr.pyx","GNN/pytorch_version/minibatch_sampler.pyx"]), include_dirs = [numpy.get_include()])
# to compile: python GNN/setup.py build_ext --inplace
