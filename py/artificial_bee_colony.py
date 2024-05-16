from typing import Callable
import torch

from common import OptimizationAlgorithm, bit_flip_n, one_max


# Note: it minimizes
class AritficialBeeColony:
    def __init__(
        self,
        fitness_fn: Callable,
        dim: int,
        n_population: int,
        domain: tuple[int, int],
        device: torch.device,
        max_trials: int = 10,
    ) -> None:
        self.device = device
        self.max_trials = max_trials
        self.fitness_fn = fitness_fn
        self.domain = domain
        self.population = torch.randint(
            self.domain[0],
            self.domain[1] + 1,
            (n_population, dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.trials = torch.zeros(n_population, dtype=torch.int32, device=self.device)
        self.fitness = torch.vmap(self.fitness_fn)(self.population)
        self.best = self.population[torch.argmin(self.fitness)]

    def _mutate_greedy_select(self, new_population):
        new_population = torch.vmap(bit_flip_n(device=self.device, n=1), randomness="different")(new_population)
        new_fitness = torch.vmap(self.fitness_fn)(new_population)
        selected = new_fitness < self.fitness
        self.trials = torch.where(selected, 0, self.trials + 1)
        self.population = torch.where(
            selected.reshape(-1, 1), new_population, self.population
        )
        self.fitness = torch.where(selected, new_fitness, self.fitness)

        best_candidate = torch.argmin(self.fitness)
        self.best = torch.where(
            self.fitness[best_candidate] < self.fitness_fn(self.best),
            self.population[best_candidate],
            self.best,
        )

    def step(self):
        # Employed bees
        self._mutate_greedy_select(self.population)

        # Onlooker bees
        fit = torch.vmap(self.fitness_fn)(self.population)
        fit = torch.where(fit >= 0, 1 / (1 + fit), 1 + torch.abs(fit))
        new_population_ind = torch.multinomial(
            fit, self.population.shape[0], replacement=True
        )
        new_population = self.population[new_population_ind]
        self._mutate_greedy_select(new_population)

        # Scout bees
        selected = self.trials >= self.max_trials
        self.population[selected] = torch.randint(
            self.domain[0],
            self.domain[1] + 1,
            (selected.sum(), self.population.shape[1]),
            device=self.device,
            dtype=torch.float32,
        )
        self.trials[selected] = 0


if __name__ == "__main__":
    abc = AritficialBeeColony(
        one_max,
        dim=500,
        n_population=100_000,
        domain=(0, 1),
        device="cuda",
    )
    for i in range(100):
        abc.step()
        print(-one_max(abc.best))
