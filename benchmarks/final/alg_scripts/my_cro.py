import sys
import src.func as func
from src.coral_reef import CoralReefOptimization


device = sys.argv[1]
dim = int(sys.argv[2])
n_population = int(sys.argv[3])

alg = CoralReefOptimization(
    device=device,
    fitness_fn=func.one_max,
    n_population=n_population,
    n_steps=1000,
    domain=(0, 1),
    dim=dim,
    settling_trials=3,
    mutation_fn=func.bit_flip_prob(),
    frac_broadcast=0.7,
    frac_duplication=0.1,
    prob_die=0.1,
)
alg.run()
result = alg.best()
print(result)
