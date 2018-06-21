# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='spykesim',
    version='0.0.0',
    description='Python module that offers functions for measuring the similarity between two segmented multi-neuronal spiking activities.',
    long_description=readme,
    author='Keita Watanabe',
    author_email='keitaw09@gmail.com',
    install_requirements=['numpy', 'cython'],
    url='https://github.com/KeitaW/spikesim',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

