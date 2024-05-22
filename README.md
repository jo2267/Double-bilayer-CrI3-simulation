[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11241392.svg)](https://doi.org/10.5281/zenodo.11241392)

# Double bilayer CrI<sub>3</sub> simulations

This repository contains a Python implementation of the following paper:

If you use this code in your research, please cite the above paper.

![](/figures/design.png)


## Hamiltonian

The code simulates spins in one Moiré superlattice with the periodic boundary condition in double bilayer CrI<sub>3</sub>. The top two layers and bottom two layers are untwisted, while the second and the third layers are twisted by angle $\theta$. Spins are on the honeycomb lattice, and the number of unit cells in one layer is $\sim (1/\theta)^2$ when $\theta$ is given in radian.

The spins on each layer are governed by a generalized Heisenberg model with single-ion anisotropy and exchange anisotropy:

$$ H^l = - \sum_{\langle i,j \rangle \in \textrm{N.N.}}  \left( J_0 \mathbf{M}_ {l,i} \cdot \mathbf{M}_ {l,j}  + K_0 M_{l, i}^z M_{l,j}^z \right) - \sum_{i}\left[ K_1 (M_{i,l}^z )^2 +  \mathbf{B} \cdot \mathbf{M} _{i,l} \right]$$ 

$\mathbf{M}$ is a vector of length $3/2$. If necessary, the asymmetry between the inner and outer layers can be introduced as a modified spin length to $3/2 \rightarrow 3/2 + dS$ for the top and bottom layers.

Spins in adjacent layers are coupled by the interlayer exchange coupling that is obtained from Figs. 2 and 3 of [Nano Lett. 18, 7658 (2018)](https://pubs.acs.org/doi/10.1021/acs.nanolett.8b03321).

$$ H_\textrm{interlayer} = -\sum_i \sum_{l=1}^{3} J^{l,l+1}_i \mathbf{M} _{l, i} \cdot \mathbf{M} _{l+1, j}$$

Only the coupling between the second and the third layer $J_{2,3}$ is spatially modulated. The other two are spatially homogeneous and antiferromagnetic 
$J_{1,2} = J_{3,4} = J^0_{\perp} < 0$. 

### Physical units

The Zeeman term in the simulation is given by $H_{Z} = - \mathbf{B} \cdot \mathbf{M} $, which is measured in meV in the code. Thus, $\mathbf{B}$ is also in the unit of meV. The actual Zeeman term is $H_{Z} = - g \mu _\textrm{B} \mathbf{M} \cdot \mathbf{B}'$ with $g \approx 2.1788$ [[PNAS 116, 1131 (2019)](https://pnas.org/doi/full/10.1073/pnas.1902100116)]. Therefore, 1 meV of $\mathbf{B}$ corresponds to $1/(g \mu _\textrm{B}) \approx 7.93$ T.

Temperature is also measured in meV, and 1 meV corresponds to 11.604 Kelvin.


## Simulations

### Input parameters
1. $\theta$: tilting angle in degree
2. $J_0$: inplane exchange coupling (meV)
3. $K_0$: exchange anisotropy (meV)
4. $K_1$: single-ion anisotropy (meV)

Other parameters can be modified in the source code. Note: fitting_harmonics = 3 is recommended to avoid overfitting for Fig. 3 data. With fitting_harmonics > 3, the fitting results critically depend on the initial condition for the least square fitting of MINPACK (i.e., scipy curve_fit). 

### Energy minimization

The total energy is minimized by the semi-implicit method combined with backtracking line search [see Chap. 3 of Exl, L. (2014). ["Tensor grid methods for micromagnetic simulations"](https://doi.org/10.34726/hss.2014.21425) ]. 


### Monte Carlo

The code can also implement Monte Carlo simulations by the Metropolis-Hastings algorithm. See LLG.py for an example of usage.

### Landau–Lifshitz–Gilbert dynamics

Real-time dynamics governed by the Landau–Lifshitz–Gilbert equation can be simulated as well. See LLG.py for an example of usage.

### Dependencies

- `scipy`
- `numpy`
- `matplotlib`

