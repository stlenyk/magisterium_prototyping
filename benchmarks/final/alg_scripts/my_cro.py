import sys
from optimization_algorithms import func
from optimization_algorithms import CoralReefOptimization

device = sys.argv[1]
dim = int(sys.argv[2])
n_population = int(sys.argv[3])
alg = CoralReefOptimization(
    device=device,
    dim=dim,
    n_population=n_population,
    n_steps=1000,
    fitness_fn=func.one_max,
    domain=(0, 1),
    settling_trials=3,
    mutation_fn=func.bit_flip_prob(),
    frac_broadcast=0.7,
    frac_duplication=0.1,
    prob_die=0.1,
)
alg.run()
result = alg.best()
print(result)
