import sys
from optimization_algorithms import func
from optimization_algorithms import ArtificialBeeColony

device = sys.argv[1]
dim = int(sys.argv[2])
n_population = int(sys.argv[3])
alg = ArtificialBeeColony(
    device=device,
    dim=dim,
    n_population=n_population,
    n_steps=1000,
    fitness_fn=func.one_max,
    domain=(0, 1),
    max_trials=10,
)
alg.run()
result = alg.best()
print(result)
