# https://github.com/VictorPelaez/coral-reef-optimization-algorithm
# remember to remove `extreme_depredation` from the `CRO` class
# and fix `multiprocess` to `multiprocessing`
import sys

import numpy as np
from cro import CRO
import numba

Ngen = 1000  # Number of generations
N = sys.argv[2]  # MxN: reef size
L = sys.argv[1]  # dim

M = 1  # MxN: reef size
Fb = 0.7  # Broadcast prob.
Fa = 0.1  # Asexual reproduction prob.
Fd = 0.1  # Fraction of the corals to be eliminated in the depredation operator.
r0 = 0.6  # Free/total initial proportion
k = 3  # Number of opportunities for a new coral to settle in the reef
Pd = 0.1  # Depredation prob.
opt = "max"  # flag: 'max' for maximizing and 'min' for minimizing
ke = 0.2


@numba.jit(nopython=True)
def max_ones(x):
    return 100 * (np.sum(x) / len(x))


cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, max_ones, opt, L, verbose=False, ke=ke)
(REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit()
