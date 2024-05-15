from typing import Callable
import torch


class CoralReef:
    def __init__(
        self,
        device: torch.device,
        fitness_fn: Callable,
        dtype: torch.dtype,
        dim: int,
        domain: tuple[torch.dtype, torch.dtype],
        mutation_range: torch.dtype,
        settling_trials: int = 10,
        frac_init_alive: float = 0.2,
        n_corals: int = 100,
        fract_broadcast: float = 0.5,
        fract_duplication: float = 0.1,
        prob_die: float = 0.5,
        mutation_fn: Callable = None,
    ):
        self.device = device
        self.fitness_fn = fitness_fn
        self.dtype = dtype
        self.domain = domain
        self.mutation_range = mutation_range
        self.settling_trials = settling_trials
        self.frac_duplication = fract_duplication
        self.prob_die = prob_die
        self.mutation_fn = mutation_fn
        self.fract_broadcast = fract_broadcast
        self.fract_duplication = fract_duplication
        self.prob_die = prob_die

        n_alive = int(frac_init_alive * n_corals)
        self.grid_alive = torch.cat(
            (
                torch.ones(n_alive, dtype=torch.bool, device=self.device),
                torch.zeros(n_corals - n_alive, dtype=torch.bool, device=self.device),
            )
        )
        self.grid_alive = self.grid_alive[torch.randperm(n_corals)]
        self.grid_values = torch.where(
            self.grid_alive.reshape(-1, 1),
            torch.randint(
                domain[0],
                domain[1] + 1,
                [n_corals, dim],
                dtype=self.dtype,
                device=self.device,
            ),
            torch.zeros([n_corals, dim], dtype=self.dtype, device=self.device),
        )

        self.grid_fitness = torch.where(
            self.grid_alive,
            torch.func.vmap(fitness_fn)(self.grid_values),
            torch.zeros(n_corals, dtype=torch.float32, device=self.device),
        )

    def _uniform_crossover(self, x, y):
        mask = torch.rand(x.shape, device=self.device) < 0.5
        return torch.where(mask, x, y)
    
    def _single_point_crossover(self, x, y):
        crossover_point = torch.randint(0, x.shape[-1], [1], device=self.device)
        mask = torch.arange(x.shape[-1], device=self.device) < crossover_point
        return torch.where(mask, x, y)
    
    def _broadcast_spawning(
        self, alive: torch.Tensor, n_broadcasters: int
    ) -> torch.Tensor:
        broadcasters = self.grid_values[alive[:n_broadcasters]]

        # crossover_points = torch.randint(
        #     0, broadcasters.shape[-1], [n_broadcasters // 2], device=self.device
        # )
        # crossover_points = crossover_points.repeat(2)
        # crossover_points = crossover_points.reshape(-1, 1)
        # crossover_points = crossover_points.repeat(1, broadcasters.shape[1])
        # broadcasters_swap = broadcasters.reshape(-1, 2, broadcasters.shape[-1])
        # broadcasters_swap = torch.flip(broadcasters_swap, [1])
        # broadcasters_swap = broadcasters_swap.reshape(-1, broadcasters.shape[-1])

        # indexing = torch.arange(broadcasters.shape[1], device=self.device)
        # indexing = indexing.reshape(1, -1)
        # indexing = indexing.repeat(broadcasters.shape[0], 1)

        # mask = indexing <= crossover_points

        # new_corals = torch.where(mask, broadcasters, broadcasters_swap)
        new_corals = torch.vmap(self._uniform_crossover, randomness="different")(
            broadcasters[::2], broadcasters[1::2]
        )

        return new_corals

    def _brooding(self, alive: torch.Tensor, n_broadcasters: int) -> torch.Tensor:
        brooders = self.grid_values[alive[n_broadcasters:]]

        if self.mutation_fn is not None:
            return self.mutation_fn(brooders, device=self.device)

        mutation = torch.randint(
            -self.mutation_range,
            self.mutation_range,
            brooders.shape,
            device=self.device,
        )
        new_corals = brooders + mutation
        new_corals = torch.clip(new_corals, self.domain[0], self.domain[1])
        return new_corals

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
            settled_mask = ~alive | (new_fitness > old_fitness)
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
        n_depradation = int(
            self.frac_duplication * torch.where(self.grid_alive)[0].shape[0]
        )
        coral_indices = torch.topk(-self.grid_fitness, n_depradation).indices
        coral_indices = coral_indices[
            torch.rand(coral_indices.shape[0], device=self.device) < self.prob_die
        ]
        self.grid_alive[coral_indices] = False

    def step(self):
        alive = torch.where(self.grid_alive)[0]
        alive = alive[torch.randperm(alive.shape[0], device=self.device)]

        n_broadcasters = int(self.fract_broadcast * alive.shape[0]) // 2 * 2
        broadcasted = self._broadcast_spawning(alive, n_broadcasters)
        brooded = self._brooding(alive, n_broadcasters)

        settling_candidates = torch.cat([broadcasted, brooded], axis=0)
        self._larvae_settling(settling_candidates)

        self._asexual_reproduction()

        self._depredation()

    def best(self):
        best_coral = self.grid_values[torch.argmax(self.grid_fitness)]
        return best_coral, self.fitness_fn(best_coral)


def bit_flip(x, device, chance=0.1):
    return torch.where(torch.rand(x.shape, device=device) < chance, 1 - x, x)


def one_max(x):
    n = x.shape[0]
    return torch.sum(x) / n * 100


torch.set_num_threads(1)


reef = CoralReef(
    device=torch.device("cpu"),
    fitness_fn=one_max,
    n_corals=1_000,
    domain=(0, 1),
    mutation_range=1,
    dim=500,
    dtype=torch.int32,
    settling_trials=3,
    mutation_fn=bit_flip,
    fract_broadcast=0.7,
    fract_duplication=0.1,
    prob_die=0.1,
)


for _ in range(100):
    reef.step()
    alive = torch.where(reef.grid_alive)[0].shape[0]
    print(f"{reef.best()[1].cpu().numpy():.2f}", alive)
