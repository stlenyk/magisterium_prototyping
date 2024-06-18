# github page: https://abcolony.github.io/
# github repository: https://github.com/abcolony/ABCPython

import datetime
import sys
import time
from ABCPython import ABC
from ABCPython import Config
import numpy as np


abc_conf = Config.Config([])  # empy argv

abc_conf.DIMENSION = int(sys.argv[1])
abc_conf.FOOD_NUMBER = int(sys.argv[2])

n_steps = 1000

# tweak, so that it doesnt't stop before n_steps
abc_conf.MAXIMUM_EVALUATION = 1_000_000_000_000
abc_conf.RUN_TIME = 1
abc_conf.SHOW_PROGRESS = False


def one_max(x):
    return -np.sum(x) / len(x) * 100


experiment_name = (
    datetime.datetime.now()
    .strftime("%Y-%m-%d %H:%M:%S")
    .replace(" ", "")
    .replace(":", "")
)
abc = ABC.ABC(abc_conf)
abc.set_experiment_id(1, experiment_name)
start_time = time.time() * 1000
abc.initial()
abc.memorize_best_source()
for _ in range(n_steps):
    abc.send_employed_bees()
    abc.calculate_probabilities()
    abc.send_onlooker_bees()
    abc.memorize_best_source()
    abc.send_scout_bees()
    abc.increase_cycle()
