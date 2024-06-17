from timeit import default_timer as timer
import numpy as np
from cmaes import CMAwM


def one_max(x):
    return np.sum(x)


def main():
    dim = 80
    optimizer = CMAwM(
        mean=np.zeros(dim),
        sigma=2.0,
        bounds=np.tile([0, 1], (dim, 1)),
        steps=np.ones(dim),
    )

    best_solution = np.zeros(dim)
    t0 = timer()
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            value = one_max(x_for_eval)
            best_solution = (
                x_for_eval if value > one_max(best_solution) else best_solution
            )
            solutions.append((x_for_tell, value))
        optimizer.tell(solutions)
        if optimizer.should_stop():
            break
        if timer() - t0 > 1:
            print(one_max(best_solution))
            t0 = timer()

    print(one_max(best_solution))


if __name__ == "__main__":
    main()
