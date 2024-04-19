# https://github.com/VictorPelaez/coral-reef-optimization-algorithm

import numpy as np
from cro import *
from cro.fitness import max_ones
from cro.report import plot_results

## ------------------------------------------------------
## Parameters initialization
## ------------------------------------------------------
Ngen = 30                  # Number of generations
N  = 20                    # MxN: reef size
M  = 20                    # MxN: reef size
Fb = 0.7                   # Broadcast prob.
Fa = 0.1                   # Asexual reproduction prob.
Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
r0 = 0.6                   # Free/total initial proportion
k  = 3                     # Number of opportunities for a new coral to settle in the reef
Pd = 0.1                   # Depredation prob.
opt= 'max'                 # flag: 'max' for maximizing and 'min' for minimizing
L = 100
ke = 0.2
## ------------------------------------------------------

cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, max_ones, opt, L, verbose=False, ke=ke)
(REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit()
# plot_results(Bestfitness, Meanfitness, cro)
