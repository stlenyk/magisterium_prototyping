import sys

from jmetal.algorithm.singleobjective import SimulatedAnnealing
from jmetal.problem import OneMax
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import BitFlipMutation


dim = int(sys.argv[1])

alg = SimulatedAnnealing(
    problem=OneMax(number_of_bits=dim),
    mutation=BitFlipMutation(probability=0.5),
    termination_criterion=StoppingByEvaluations(1_000),
)

alg.run()
res = sum(alg.get_result().variables[0]) / dim * 100
print(res)
