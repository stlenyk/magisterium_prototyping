# Optimization algoritms with PyTorch

A repository containing all tests (directory `benchmarks`) done during the development of Master's thesis: _Optimization Algorithms for Discrete Problems using Tensor Computations_ along with the library created as a result (directory `optimization_algorithms`).

## Installation

```sh
pip install "git+https://github.com/stlenyk/magisterium_prototyping/#egg=optimization_algorithms&subdirectory=optimization_algorithms" 
```

## Usage

This small library includes algorihm modules:

- [`CoralReefOptimization`](optimization_algorithms/optimization_algorithms/coral_reef.py) ([paper](https://doi.org/10.1155/2014/739768))
- [`ArtificialBeeColony`](optimization_algorithms/optimization_algorithms/artificial_bee_colony.py)
- [`SimulatedAnnealing`](optimization_algorithms/optimization_algorithms/simulated_annealing.py)

and a [`func`](optimization_algorithms/optimization_algorithms/func.py) module with some predefined fitness (cost) functions, mutation and crossover functions.

Example:

```py
from optimization_algorithms import CoralReefOptimization, func

alg = CoralReefOptimization(
    device="cuda",  # or "cpu"
    dim=500,
    n_population=1000,
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
```
