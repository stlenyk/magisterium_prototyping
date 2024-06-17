import sys
from optimization_algorithms import func
from optimization_algorithms import SimulatedAnnealing

device = sys.argv[1]
dim = int(sys.argv[2])
alg = SimulatedAnnealing(
    device=device,
    dim=dim,
    n_steps=1_000,
    fitness_fn=func.one_max,
    domain=(0, 1),
    mutation_fn=func.bit_flip_prob(p=0.5),
)
alg.run()
result = alg.best()
print(result)
