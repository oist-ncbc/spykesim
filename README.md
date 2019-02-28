# Spykesim
![PyPI](https://img.shields.io/pypi/v/spykesim.svg)

spykesim is a Python module that offers functions for measuring the similarity between two segmented multi-neuronal spiking activities.
Extended edit similarity measurement is implemented. You can find the details in the following paper.
bioArxiv: https://www.biorxiv.org/content/early/2017/10/30/202655
# Supported Operating Systems
Ubuntu and MacOS. For Windows users: Please consider to use Ubuntu via Windows Subsystem for Linux.

# Installation
You can install via pip.
```python
pip install spykesim
```

## Dependencies

- Python (>= 3.5)
- Cython
- Numpy
- tqdm
Cython and Numpy needs to be preinstalled as these will be used in the installation process.
# Tutorial 
You can find a tutorial in [doc](https://github.com/KeitaW/spykesim/blob/master/docs/tutorial.ipynb).

# Citation
You can use the following bib entry to cite this work:
```
@article{Watanabe:2017bla,
author = {Watanabe, Keita and Haga, Tatsuya and Euston, David R and Tatsuno, Masami and Fukai, Tomoki},
title = {{Unsupervised detection of cell-assembly sequences with edit similarity score}},
year = {2017},
pages = {202655},
month = oct
}
```

# 


This project uses the following repository as a template.

https://github.com/kennethreitz/samplemod 
Copyright (c) 2017, Kenneth Reitz
