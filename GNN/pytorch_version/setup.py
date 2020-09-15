from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "GNN.pytorch_version.minibatch_sampler",
        ["GNN/pytorch_version/minibatch_sampler.pyx", "GNN/pytorch_version/sampler_core.cpp"],
        language="c++",
        extra_compile_args=['-fopenmp','-O3', '-std=c++11'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),include_dirs=[numpy.get_include()]
)