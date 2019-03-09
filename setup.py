# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os
from glob import glob

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()
try:
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools import Extension
    USING_CYTHON = False
else:
    USING_CYTHON = True
print("using cython: ", USING_CYTHON)
ext = 'pyx' if USING_CYTHON else 'c'
sources = glob('spykesim/*.%s' % (ext,))
extensions = [
    Extension(source.split('.')[0].replace(os.path.sep, '.'),
              sources=[source],
    )
for source in sources]
cmdclass = {'build_ext': build_ext} if USING_CYTHON else {}

import numpy
setup(
    name='spykesim',
    version='1.0.0',
    description='Python module that offers functions for measuring the similarity between two segmented multi-neuronal spiking activities.',
    long_description=readme,
    author='Keita Watanabe',
    author_email='keitaw09@gmail.com',
    install_requires=['scipy', 'joblib', 'tqdm'],
    url='https://github.com/KeitaW/spikesim',
    license=license,
    ext_modules=extensions,
    cmdclass=cmdclass,
    include_dirs = [numpy.get_include()],
)

