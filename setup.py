# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='spykesim',
    version='0.0.0',
    description='Similarity mesurement between two segmented multi-neuronal spiking activities.',
    long_description=readme,
    author='Keita Watanabe',
    author_email='keitaw09@gmail.com',
    url='https://github.com/KeitaW/spikesim',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

