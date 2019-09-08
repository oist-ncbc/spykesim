# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os
from glob import glob

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext
ext = 'pyx'
sources = glob('spykesim/*.%s' % (ext,))
extensions = [
    Extension(source.split('.')[0].replace(os.path.sep, '.'),
              sources=[source],
    )
for source in sources]
cmdclass = {'build_ext': build_ext}

import numpy
setup(
    name='spykesim',
    version='1.2.0',
    description='Python module that offers functions for measuring the similarity between two segmented multi-neuronal spiking activities.',
    long_description=readme,
    author='Keita Watanabe',
    author_email='keitaw09@gmail.com',
    install_requires=['scipy', 'joblib', 'tqdm', 'h5py'],
    url='https://github.com/KeitaW/spikesim',
    license=license,
    ext_modules=extensions,
    cmdclass=cmdclass,
    include_dirs = [numpy.get_include()],
)

