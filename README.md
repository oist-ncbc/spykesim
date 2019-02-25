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

# Tutorial 

# Algorithms

# Similarity measures
This project offers variants of edit similarity measurement, simple, linear gap penalty, local alignment with linear gap penalty, and local alignment with exponentially growing gap penalty.

# Simple
Let <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon_{j}^{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon_{j}^{i}" title="\epsilon_{j}^{i}" /></a> be the number of partial coincidences obtained up to the i-th element of <a href="https://www.codecogs.com/eqnedit.php?latex=W(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W(1)" title="W(1)" /></a> and the j-th element of <a href="https://www.codecogs.com/eqnedit.php?latex=W(2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W(2)" title="W(2)" /></a>$W(2)$.

<a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon_{j}^{i}&space;=&space;\max&space;\begin{cases}&space;\epsilon_{j}^{i&space;-&space;1}&space;\\&space;\epsilon_{j&space;-&space;1}^{i}&space;\\&space;\epsilon_{j&space;-&space;1}^{i&space;-&space;1}&space;&plus;&space;\delta(W(1)\lbrack&space;j\rbrack,W(2)\lbrack&space;i\rbrack)&space;\\&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon_{j}^{i}&space;=&space;\max&space;\begin{cases}&space;\epsilon_{j}^{i&space;-&space;1}&space;\\&space;\epsilon_{j&space;-&space;1}^{i}&space;\\&space;\epsilon_{j&space;-&space;1}^{i&space;-&space;1}&space;&plus;&space;\delta(W(1)\lbrack&space;j\rbrack,W(2)\lbrack&space;i\rbrack)&space;\\&space;\end{cases}" title="\epsilon_{j}^{i} = \max \begin{cases} \epsilon_{j}^{i - 1} \\ \epsilon_{j - 1}^{i} \\ \epsilon_{j - 1}^{i - 1} + \delta(W(1)\lbrack j\rbrack,W(2)\lbrack i\rbrack) \\ \end{cases}" /></a>
<!--
$$
  \epsilon_{j}^{i} = \max
  \begin{cases}
  \epsilon_{j}^{i - 1} \\
  \epsilon_{j - 1}^{i} \\
  \epsilon_{j - 1}^{i - 1} + \delta(W(1)\lbrack j\rbrack,W(2)\lbrack i\rbrack) \\ 
  \end{cases}
$$
-->

# Local alignment (Not yet implmented)

# Local alignment with linear gap penalty (Not yet implmented)

# Local alignment with exponentially growing gap penalty (Not yet implmented)

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\upsilon_{j}^{i}&space;&=&&space;\begin{cases}&space;1&space;&&space;\epsilon_{j}^{i&space;-&space;1}&space;-&space;\exp\left(&space;\alpha&space;\right)&space;\geq&space;\epsilon_{j}^{i&space;-&space;1&space;-&space;\upsilon_{j}^{i&space;-&space;1}}&space;-&space;\exp\left(&space;\alpha\upsilon_{j}^{i&space;-&space;1}&space;\right)&space;\\&space;\upsilon_{j}^{i&space;-&space;1}&space;&plus;&space;1&space;&&space;\text{otherwise}&space;\\&space;\end{cases}&space;\\&space;\rho_{j}^{i}&space;&=&&space;\begin{cases}&space;1&space;&&space;\epsilon_{j&space;-&space;1}^{i}&space;-&space;\exp\left(&space;\alpha&space;\right)&space;\geq&space;\epsilon_{j&space;-&space;1&space;-&space;\rho_{j&space;-&space;1}^{i}}^{i}&space;-&space;\exp\left(&space;\alpha\rho_{j&space;-&space;1}^{i}&space;\right)&space;\\&space;\rho_{j&space;-&space;1}^{i}&space;&plus;&space;1&space;&&space;\text{otherwise}&space;\\&space;\end{cases}&space;\\&space;\epsilon_{j}^{i}&space;&=&&space;\max&space;\begin{cases}&space;0&space;\\&space;\epsilon_{j}^{i&space;-&space;\upsilon_{j}^{i&space;-&space;1}}&space;-&space;\exp(a\upsilon_{j}^{i&space;-&space;1})&space;\\&space;\epsilon_{j&space;-&space;\rho_{j&space;-&space;1}^{i}}^{i}&space;-&space;\exp(a\rho_{j&space;-&space;1}^{i})&space;\\&space;\epsilon_{j&space;-&space;1}^{i&space;-&space;1}&space;&plus;&space;\mathbf{r}_{i}(t_{k})&space;\cdot&space;\mathbf{r}_{j}(t_{k'})&space;\\&space;\end{cases}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\upsilon_{j}^{i}&space;&=&&space;\begin{cases}&space;1&space;&&space;\epsilon_{j}^{i&space;-&space;1}&space;-&space;\exp\left(&space;\alpha&space;\right)&space;\geq&space;\epsilon_{j}^{i&space;-&space;1&space;-&space;\upsilon_{j}^{i&space;-&space;1}}&space;-&space;\exp\left(&space;\alpha\upsilon_{j}^{i&space;-&space;1}&space;\right)&space;\\&space;\upsilon_{j}^{i&space;-&space;1}&space;&plus;&space;1&space;&&space;\text{otherwise}&space;\\&space;\end{cases}&space;\\&space;\rho_{j}^{i}&space;&=&&space;\begin{cases}&space;1&space;&&space;\epsilon_{j&space;-&space;1}^{i}&space;-&space;\exp\left(&space;\alpha&space;\right)&space;\geq&space;\epsilon_{j&space;-&space;1&space;-&space;\rho_{j&space;-&space;1}^{i}}^{i}&space;-&space;\exp\left(&space;\alpha\rho_{j&space;-&space;1}^{i}&space;\right)&space;\\&space;\rho_{j&space;-&space;1}^{i}&space;&plus;&space;1&space;&&space;\text{otherwise}&space;\\&space;\end{cases}&space;\\&space;\epsilon_{j}^{i}&space;&=&&space;\max&space;\begin{cases}&space;0&space;\\&space;\epsilon_{j}^{i&space;-&space;\upsilon_{j}^{i&space;-&space;1}}&space;-&space;\exp(a\upsilon_{j}^{i&space;-&space;1})&space;\\&space;\epsilon_{j&space;-&space;\rho_{j&space;-&space;1}^{i}}^{i}&space;-&space;\exp(a\rho_{j&space;-&space;1}^{i})&space;\\&space;\epsilon_{j&space;-&space;1}^{i&space;-&space;1}&space;&plus;&space;\mathbf{r}_{i}(t_{k})&space;\cdot&space;\mathbf{r}_{j}(t_{k'})&space;\\&space;\end{cases}&space;\end{align*}" title="\begin{align*} \upsilon_{j}^{i} &=& \begin{cases} 1 & \epsilon_{j}^{i - 1} - \exp\left( \alpha \right) \geq \epsilon_{j}^{i - 1 - \upsilon_{j}^{i - 1}} - \exp\left( \alpha\upsilon_{j}^{i - 1} \right) \\ \upsilon_{j}^{i - 1} + 1 & \text{otherwise} \\ \end{cases} \\ \rho_{j}^{i} &=& \begin{cases} 1 & \epsilon_{j - 1}^{i} - \exp\left( \alpha \right) \geq \epsilon_{j - 1 - \rho_{j - 1}^{i}}^{i} - \exp\left( \alpha\rho_{j - 1}^{i} \right) \\ \rho_{j - 1}^{i} + 1 & \text{otherwise} \\ \end{cases} \\ \epsilon_{j}^{i} &=& \max \begin{cases} 0 \\ \epsilon_{j}^{i - \upsilon_{j}^{i - 1}} - \exp(a\upsilon_{j}^{i - 1}) \\ \epsilon_{j - \rho_{j - 1}^{i}}^{i} - \exp(a\rho_{j - 1}^{i}) \\ \epsilon_{j - 1}^{i - 1} + \mathbf{r}_{i}(t_{k}) \cdot \mathbf{r}_{j}(t_{k'}) \\ \end{cases} \end{align*}" /></a>
<!--
$$
\begin{align*} 
\upsilon_{j}^{i} &=& \begin{cases}
1 & \epsilon_{j}^{i - 1} - \exp\left( \alpha \right) \geq \epsilon_{j}^{i - 1 - \upsilon_{j}^{i - 1}} - \exp\left( \alpha\upsilon_{j}^{i - 1} \right) \\
\upsilon_{j}^{i - 1} + 1 & \text{otherwise} \\
\end{cases} \\
\rho_{j}^{i} &=& \begin{cases}
1 & \epsilon_{j - 1}^{i} - \exp\left( \alpha \right) \geq \epsilon_{j - 1 - \rho_{j - 1}^{i}}^{i} - \exp\left( \alpha\rho_{j - 1}^{i} \right) \\
\rho_{j - 1}^{i} + 1 & \text{otherwise} \\
\end{cases} \\
\epsilon_{j}^{i} &=& \max \begin{cases}
0 \\
\epsilon_{j}^{i - \upsilon_{j}^{i - 1}} - \exp(a\upsilon_{j}^{i - 1}) \\
\epsilon_{j - \rho_{j - 1}^{i}}^{i} - \exp(a\rho_{j - 1}^{i}) \\
\epsilon_{j - 1}^{i - 1} + \mathbf{r}_{i}(t_{k}) \cdot \mathbf{r}_{j}(t_{k'}) \\
\end{cases}
\end{align*}
$$
-->


<font size="3">
This project uses the following repository as a template.
https://github.com/kennethreitz/samplemod 
Copyright (c) 2017, Kenneth Reitz

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
</font>
