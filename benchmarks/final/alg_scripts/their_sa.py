import sys

from jmetal.algorithm.singleobjective import SimulatedAnnealing
from jmetal.problem import OneMax
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import BitFlipMutation


dim = sys.argv[1]

algorithm = SimulatedAnnealing(
    problem=OneMax(number_of_bits=dim),
    mutation=BitFlipMutation(probability=0.5),
    termination_criterion=StoppingByEvaluations(1_000),
)

algorithm.run()
res = sum(algorithm.get_result().variables[0]) / dim * 100
print(f"{res:.2f}")
