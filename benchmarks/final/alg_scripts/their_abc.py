# github page: https://abcolony.github.io/
# github repository: https://github.com/abcolony/ABCPython


import sys
from ABCPython import ABC, Config
import numpy as np


def one_max(x):
    return (-np.sum(x) / len(x) * 100,)


abc_conf = Config.Config([])  # empy argv

abc_conf.DIMENSION = int(sys.argv[1])
abc_conf.FOOD_NUMBER = int(sys.argv[2])
abc_conf.LOWER_BOUND = 0
abc_conf.UPPER_BOUND = 1
abc_conf.OBJECTIVE_FUNCTION = one_max
abc_conf.LIMIT = 10

# tweak, so that it doesnt't stop before n_steps
abc_conf.MAXIMUM_EVALUATION = 1_000_000_000_000
abc_conf.RUN_TIME = 1
abc_conf.SHOW_PROGRESS = False

n_steps = 1000

abc = ABC.ABC(abc_conf)
abc.initial()
abc.memorize_best_source()
for _ in range(n_steps):
    abc.send_employed_bees()
    abc.calculate_probabilities()
    abc.send_onlooker_bees()
    abc.memorize_best_source()
    abc.send_scout_bees()
    abc.increase_cycle()

print(-abc.globalOpt)
