from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "graphsaint.pytorch_version.minibatch_sampler",
        ["graphsaint/pytorch_version/minibatch_sampler.pyx", "graphsaint/pytorch_version/sampler_core.cpp"],
        language="c++",
        extra_compile_args=['-fopenmp','-O3'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),include_dirs=[numpy.get_include()]
)