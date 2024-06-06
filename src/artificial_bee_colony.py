import common

import torch


# Note: it minimizes
class ArtificialBeeColony:
    def __init__(
        self,
        fitness_fn: common.TensorFn,
        dim: int,
        domain: common.Domain,
        dtype: torch.dtype = torch.float32,
        n_population: int = 100,
        max_trials: int = 10,
        mutation_fn: common.TensorFn = None,
        device: torch.device = "cpu",
    ) -> None:
        self.device = device
        self.fitness_fn = fitness_fn
        self.domain = domain
        self.dtype = dtype
        self.max_trials = max_trials
        if mutation_fn is None:
            mutation_fn = common.mutation_n(
                domain_clip=domain,
                draw_range=domain,
                n=1,
            )
        self.mutation_fn = mutation_fn
        self.population = torch.randint(
            self.domain[0],
            self.domain[1] + 1,
            (n_population, dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.trials = torch.zeros(n_population, dtype=torch.int32, device=self.device)
        self.best = self.population[
            torch.argmin(torch.vmap(self.fitness_fn)(self.population))
        ]

    def _mutate_greedy_select(self, new_population):
        new_population = torch.vmap(self.mutation_fn, randomness="different")(
            new_population
        )
        old_fitness = torch.vmap(self.fitness_fn)(self.population)
        new_fitness = torch.vmap(self.fitness_fn)(new_population)
        selected = new_fitness < old_fitness
        self.trials = torch.where(selected, 0, self.trials + 1)
        self.population = torch.where(
            selected.reshape(-1, 1), new_population, self.population
        )

    def _update_best(self, new_fitness):
        best_candidate = torch.argmin(new_fitness)
        self.best = torch.where(
            new_fitness[best_candidate] < self.fitness_fn(self.best),
            self.population[best_candidate],
            self.best,
        )

    # Notice how steps 1. (employed bees) and 2. (onlooker bees) could be combined
    # by removing 1. and adding in 2. one more candidate from each existing population member
    def step(self):
        # 1. Employed bees
        self._mutate_greedy_select(self.population)

        # 2. Onlooker bees
        fit = torch.vmap(self.fitness_fn)(self.population)
        fit = torch.where(fit >= 0, 1 / (1 + fit), 1 + torch.abs(fit))
        new_population_ind = torch.multinomial(
            fit, self.population.shape[0], replacement=True
        )
        new_population = self.population[new_population_ind]
        new_population = torch.vmap(self.mutation_fn, randomness="different")(
            new_population
        )
        new_fitness = torch.vmap(self.fitness_fn)(new_population)
        # sort descending, so that when indexing later, smallest (best) fitnesses and consequently solutions are chosen last,
        # i.e. they will be the ones that are inserted into the tensor
        new_fitness, indices = torch.sort(new_fitness)
        new_population = new_population[indices]
        new_population_ind = new_population_ind[indices]

        old_fitness = torch.vmap(self.fitness_fn)(self.population)
        selected = new_fitness < old_fitness[new_population_ind]
        self.population[new_population_ind] = torch.where(
            selected.reshape(-1, 1), new_population, self.population[new_population_ind]
        )
        self.trials[new_population_ind] = torch.where(
            selected, 0, self.trials[new_population_ind] + 1
        )

        # 3. Scout bees
        selected = self.trials > self.max_trials
        self.population[selected] = torch.randint(
            self.domain[0],
            self.domain[1] + 1,
            (selected.sum(), self.population.shape[1]),
            dtype=self.dtype,
            device=self.population.device,
        )
        self.trials[selected] = 0

        self._update_best(torch.vmap(self.fitness_fn)(self.population))


if __name__ == "__main__":
    abc = ArtificialBeeColony(
        common.one_max,
        dim=500,
        n_population=100_000,
        domain=(0, 1),
        mutation_fn=common.bit_flip_n(100),
        device="cuda",
    )

    from timeit import default_timer as timer

    t0 = timer()
    for i in range(100):
        abc.step()
        print(-common.one_max(abc.best))
    elapsed = timer() - t0
    print(f"Elapsed: {elapsed:.2f}s")
