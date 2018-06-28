from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy
Cython.Compiler.Options.annotate = True

setup(
    name = "editsim",
    ext_modules=cythonize("editsim.pyx"),
    include_dirs = [numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)
