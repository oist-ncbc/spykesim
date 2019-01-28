# Spykesim

spykesim is a Python module that offers functions for measuring the similarity between two segmented multi-neuronal spiking activities.
Extended edit similarity measurement is implemented. You can find the details in the following paper.
PAPERCITATION

# Installation

## Dependencies

- Python (>= 3.4)

## User installation

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
