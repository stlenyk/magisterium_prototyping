import common

import torch


class CoralReef:
    def __init__(
        self,
        fitness_fn: common.TensorFn,
        dim: int,
        domain: common.Domain,
        dtype: torch.dtype = None,
        n_population: int = 100,
        settling_trials: int = 10,
        frac_init_alive: float = 0.2,
        frac_broadcast: float = 0.5,
        frac_duplication: float = 0.1,
        prob_die: float = 0.5,
        mutation_fn: common.TensorFn = None,
        device: torch.device = "cpu",
    ):
        self.device = device
        self.fitness_fn = fitness_fn
        self.dtype = dtype
        self.domain = domain
        self.settling_trials = settling_trials
        self.frac_duplication = frac_duplication
        self.prob_die = prob_die
        if mutation_fn is None:
            mutation_fn = common.mutation_prob(
                domain_clip=domain, draw_range=domain, probability=0.1
            )
        self.mutation_fn = mutation_fn
        self.fract_broadcast = frac_broadcast
        self.fract_duplication = frac_duplication
        self.prob_die = prob_die

        n_alive = int(frac_init_alive * n_population)
        self.grid_alive = torch.cat(
            (
                torch.ones(n_alive, dtype=torch.bool, device=self.device),
                torch.zeros(
                    n_population - n_alive, dtype=torch.bool, device=self.device
                ),
            )
        )
        self.grid_alive = self.grid_alive[torch.randperm(n_population)]
        self.grid_values = torch.where(
            self.grid_alive.reshape(-1, 1),
            torch.randint(
                domain[0],
                domain[1] + 1,
                [n_population, dim],
                dtype=self.dtype,
                device=self.device,
            ),
            torch.zeros([n_population, dim], dtype=self.dtype, device=self.device),
        )

        self.grid_fitness = torch.where(
            self.grid_alive,
            torch.func.vmap(fitness_fn)(self.grid_values),
            torch.zeros(n_population, dtype=torch.float32, device=self.device),
        )

    def _broadcast_spawning(
        self, alive: torch.Tensor, n_broadcasters: int
    ) -> torch.Tensor:
        broadcasters = self.grid_values[alive[:n_broadcasters]]
        new_corals = torch.vmap(common.uniform_crossover, randomness="different")(
            broadcasters[::2], broadcasters[1::2]
        )

        return new_corals

    def _brooding(self, alive: torch.Tensor, n_broadcasters: int) -> torch.Tensor:
        brooders = self.grid_values[alive[n_broadcasters:]]
        return torch.vmap(self.mutation_fn, randomness="different")(brooders)

    def _larvae_settling(self, new_corals: torch.Tensor):
        for _ in range(self.settling_trials):
            if len(new_corals) == 0:
                break

            indices = torch.randperm(self.grid_values.shape[0], device=self.device)[
                : new_corals.shape[0]
            ]

            alive = self.grid_alive[indices]
            old_fitness = self.grid_fitness[indices]
            new_fitness = torch.func.vmap(self.fitness_fn)(new_corals)
            settled_mask = ~alive | (new_fitness < old_fitness)
            settled_indices = indices[settled_mask]

            self.grid_values[settled_indices] = new_corals[settled_mask]
            self.grid_fitness[settled_indices] = new_fitness[settled_mask]
            self.grid_alive[settled_indices] = True

            new_corals = new_corals[~settled_mask]

    def _asexual_reproduction(self):
        n_duplication = int(
            self.fract_duplication * torch.where(self.grid_alive)[0].shape[0]
        )
        best_corals = self.grid_values[
            torch.topk(self.grid_fitness, n_duplication).indices
        ]
        self._larvae_settling(best_corals)

    def _depredation(self):
        n_depredation = int(
            self.frac_duplication * torch.where(self.grid_alive)[0].shape[0]
        )
        coral_indices = torch.topk(-self.grid_fitness, n_depredation).indices
        coral_indices = coral_indices[
            torch.rand(coral_indices.shape[0], device=self.device) < self.prob_die
        ]
        self.grid_alive[coral_indices] = False

    def step(self):
        """Perform a single step of the CRO algorithm."""
        alive = torch.where(self.grid_alive)[0]
        alive = alive[torch.randperm(alive.shape[0], device=self.device)]

        n_broadcasters = int(self.fract_broadcast * alive.shape[0]) // 2 * 2
        broadcasted = self._broadcast_spawning(alive, n_broadcasters)
        brooded = self._brooding(alive, n_broadcasters)

        settling_candidates = torch.cat([broadcasted, brooded], dim=0)
        self._larvae_settling(settling_candidates)

        self._asexual_reproduction()

        self._depredation()

    def best(self):
        best_coral = self.grid_values[torch.argmin(self.grid_fitness)]
        return best_coral, self.fitness_fn(best_coral)


if __name__ == "__main__":

    reef = CoralReef(
        device="cuda",
        fitness_fn=common.one_max,
        n_population=100_000,
        domain=(0, 1),
        dim=500,
        settling_trials=3,
        mutation_fn=common.bit_flip_prob(),
        frac_broadcast=0.7,
        frac_duplication=0.1,
        prob_die=0.1,
    )

    for _ in range(100):
        reef.step()
        alive = torch.where(reef.grid_alive)[0].shape[0]
        print(f"{-reef.best()[1].cpu().numpy():.2f}", alive)
