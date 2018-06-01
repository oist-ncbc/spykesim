# (WIP) spykesim

spykesim is a Python module that offers functions for measuring the similarity between two segmented multi-neuronal spiking activities.
Extended edit similarity measurement is implemented. You can find the details in the following paper.
PAPERCITATION

# Installation

## Dependencies

- Python (>= 3.4)

## User installation


# Tutorial
This project offers four variants of edit similarity measurement, simple, linear gap penalty, local alignment with linear gap penalty, local alignment with exponentially growing gap penalty.

# Simple
Let $\epsilon_{j}^{i}$ be the number of partial coincidences obtained up to the $i\mathchar`-\mathrm{th}$ element of $W(1)$ and the $j\mathchar`-\mathrm{th}$ element of $W(2)$.
$$
  \epsilon_{j}^{i} = \max
  \begin{cases}
  \epsilon_{j}^{i - 1} \\
  \epsilon_{j - 1}^{i} \\
  \epsilon_{j - 1}^{i - 1} + \delta(W(1)\lbrack j\rbrack,W(2)\lbrack i\rbrack) \\ 
  \end{cases}
$$

# Local alignment (Not yet implmented)

# Local alignment with linear gap penalty (Not yet implmented)

# Local alignment with exponentially growing gap penalty (Not yet implmented)
$$
\upsilon_{j}^{i} = \begin{cases}
1 & \epsilon_{j}^{i - 1} - \exp\left( \alpha \right) \geq \epsilon_{j}^{i - 1 - \upsilon_{j}^{i - 1}} - \exp\left( \alpha\upsilon_{j}^{i - 1} \right) \\
\upsilon_{j}^{i - 1} + 1 & \text{otherwise} \\
\end{cases}, \\
\rho_{j}^{i} = \begin{cases}
1 & \epsilon_{j - 1}^{i} - \exp\left( \alpha \right) \geq \epsilon_{j - 1 - \rho_{j - 1}^{i}}^{i} - \exp\left( \alpha\rho_{j - 1}^{i} \right) \\
\rho_{j - 1}^{i} + 1 & \text{otherwise} \\
\end{cases},
$$
$$
\epsilon_{j}^{i} = \max \begin{cases}
0 \\
\epsilon_{j}^{i - \upsilon_{j}^{i - 1}} - \exp(a\upsilon_{j}^{i - 1}) \\
\epsilon_{j - \rho_{j - 1}^{i}}^{i} - \exp(a\rho_{j - 1}^{i}) \\
\epsilon_{j - 1}^{i - 1} + \mathbf{r}_{i}(t_{k}) \cdot \mathbf{r}_{j}(t_{k'}) \\
\end{cases},
$$
