'''

import tensorflow as tf
from typing import Callable


class _CoralReef:
    """
    Note: it maximizes
    """

    def __init__(
        self,
        fitness_fn: Callable,
        dtype: tf.dtypes.DType,
        dim: int,
        domain: tuple[tf.dtypes.DType, tf.dtypes.DType],
        mutation_range: tf.dtypes.DType,
        settling_trials: int = 10,
        frac_init_alive: float = 0.2,
        n_corals: int = 100,
        fract_broadcast: float = 0.5,
        fract_duplication: float = 0.1,
        prob_die: float = 0.5,
        mutation_fn: Callable = None,
    ):
        self.fitness_fn = fitness_fn
        self.dtype = dtype
        self.grid_values = tf.zeros([n_corals, dim], dtype=self.dtype)
        self.grid_fitness = tf.zeros(n_corals, dtype=tf.float32)
        self.grid_alive = tf.zeros(n_corals, dtype=tf.bool)
        self.domain = domain
        self.mutation_range = mutation_range
        self.settling_trials = settling_trials
        self.frac_duplication = fract_duplication
        self.prob_die = prob_die
        self.mutation_fn = mutation_fn

        alive_ind = tf.random.shuffle(tf.range(n_corals))[
            : int(frac_init_alive * n_corals)
        ]
        alive_ind = tf.reshape(alive_ind, [-1, 1])
        self.grid_values = tf.tensor_scatter_nd_update(
            self.grid_values,
            alive_ind,
            tf.random.uniform(
                [len(alive_ind), dim],
                minval=domain[0],
                maxval=domain[1] + 1,
                dtype=self.dtype,
            ),
        )
        self.grid_fitness = tf.tensor_scatter_nd_update(
            self.grid_fitness,
            alive_ind,
            tf.vectorized_map(
                fitness_fn,
                tf.gather(self.grid_values, alive_ind),
            ),
        )
        self.grid_alive = tf.tensor_scatter_nd_update(
            self.grid_alive, alive_ind, tf.ones(len(alive_ind), dtype=tf.bool)
        )
        self.fract_broadcast = fract_broadcast
        self.fract_duplication = fract_duplication
        self.prob_die = prob_die

    def _broadcast_spawning(self, alive: tf.Tensor, n_broadcasters: int) -> tf.Tensor:
        broadcasters = tf.gather(self.grid_values, alive[:n_broadcasters])

        crossover_points = tf.random.uniform(
            [n_broadcasters // 2],
            minval=0,
            maxval=broadcasters.shape[-1],
            dtype=tf.int32,
        )
        crossover_points = tf.repeat(crossover_points, 2)
        crossover_points = tf.reshape(crossover_points, [-1, 1])
        crossover_points = tf.repeat(crossover_points, broadcasters.shape[1], axis=1)
        broadcasters_swap = tf.reshape(broadcasters, [-1, 2, broadcasters.shape[-1]])
        broadcasters_swap = tf.reverse(broadcasters_swap, [1])
        broadcasters_swap = tf.reshape(broadcasters_swap, [-1, broadcasters.shape[-1]])

        indexing = tf.range(broadcasters.shape[1])
        indexing = tf.reshape(indexing, [1, -1])
        indexing = tf.repeat(indexing, broadcasters.shape[0], axis=0)

        mask = indexing <= crossover_points

        new_corals = tf.where(mask, broadcasters, broadcasters_swap)
        return new_corals

    def _brooding(self, alive: tf.Tensor, n_broadcasters: int) -> tf.Tensor:
        brooders = tf.gather(self.grid_values, alive[n_broadcasters:])

        if self.mutation_fn is not None:
            return self.mutation_fn(brooders)

        mutation = tf.random.uniform(
            brooders.shape,
            minval=-self.mutation_range,
            maxval=self.mutation_range,
            dtype=self.dtype,
        )
        new_corals = brooders + mutation
        new_corals = tf.clip_by_value(new_corals, self.domain[0], self.domain[1])
        return new_corals

    def _larvae_settling(self, new_corals: tf.Tensor):
        for _ in range(self.settling_trials):
            if len(new_corals) == 0:
                break

            indices = tf.random.shuffle(tf.range(len(self.grid_values)))[
                : len(new_corals)
            ]

            alive = tf.gather(self.grid_alive, indices)
            old_fitness = tf.gather(self.grid_fitness, indices)
            new_fitness = tf.vectorized_map(self.fitness_fn, new_corals)
            settled_mask = ~alive | (new_fitness > old_fitness)
            settled_indices = tf.reshape(
                tf.boolean_mask(indices, settled_mask), [-1, 1]
            )

            self.grid_values = tf.tensor_scatter_nd_update(
                self.grid_values,
                settled_indices,
                tf.boolean_mask(new_corals, settled_mask),
            )
            self.grid_fitness = tf.tensor_scatter_nd_update(
                self.grid_fitness,
                settled_indices,
                tf.boolean_mask(new_fitness, settled_mask),
            )
            self.grid_alive = tf.tensor_scatter_nd_update(
                self.grid_alive, settled_indices, tf.repeat(True, len(settled_indices))
            )

            new_corals = tf.boolean_mask(new_corals, ~settled_mask)

    def _asexual_reproduction(self):
        n_duplication = int(self.fract_duplication * len(tf.where(self.grid_alive)))
        best_corals = tf.gather(
            self.grid_values, tf.math.top_k(self.grid_fitness, n_duplication).indices
        )
        self._larvae_settling(best_corals)

    def _depredation(self):
        n_depradation = int(self.frac_duplication * len(tf.where(self.grid_alive)))
        coral_indices = tf.math.top_k(-self.grid_fitness, n_depradation).indices
        coral_indices = tf.boolean_mask(
            coral_indices,
            tf.random.uniform([coral_indices.shape[0]]) < self.prob_die,
        )
        self.grid_alive = tf.tensor_scatter_nd_update(
            self.grid_alive,
            tf.reshape(coral_indices, [-1, 1]),
            tf.repeat(False, len(coral_indices)),
        )

    def step(self):
        alive = tf.reshape(tf.where(self.grid_alive), [-1])
        tf.random.shuffle(alive)

        n_broadcasters = int(self.fract_broadcast * len(alive)) // 2 * 2
        broadcasted = self._broadcast_spawning(alive, n_broadcasters)
        brooded = self._brooding(alive, n_broadcasters)

        settling_candidates = tf.concat([broadcasted, brooded], axis=0)
        self._larvae_settling(settling_candidates)

        self._asexual_reproduction()

        self._depredation()

    def best(self):
        best_coral = tf.gather(self.grid_values, tf.math.argmax(self.grid_fitness))
        return best_coral, self.fitness_fn(best_coral)


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def bit_flip(x, chance=0.1):
    return tf.where(tf.random.uniform(x.shape) < chance, 1 - x, x)


@tf.function
def one_max(x):
    n = x.shape[0]
    return tf.cast(tf.reduce_sum(x) / n * 100, dtype=tf.float32)
'''

import torch
from typing import Callable

class CoralReef:
    def __init__(
        self,
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
        self.fitness_fn = fitness_fn
        self.dtype = dtype
        self.grid_values = torch.zeros([n_corals, dim], dtype=self.dtype)
        self.grid_fitness = torch.zeros(n_corals, dtype=torch.float32)
        self.grid_alive = torch.zeros(n_corals, dtype=torch.bool)
        self.domain = domain
        self.mutation_range = mutation_range
        self.settling_trials = settling_trials
        self.frac_duplication = fract_duplication
        self.prob_die = prob_die
        self.mutation_fn = mutation_fn

        alive_ind = torch.randperm(n_corals)[: int(frac_init_alive * n_corals)]
        alive_ind = alive_ind.reshape(-1, 1)
        # Make grid_values have random vectors at alive indices and zeros elsewhere
        self.grid_values = torch.tensor(
            self.grid_values.scatter(0, alive_ind, torch.randint(
                domain[0], domain[1] + 1, [len(alive_ind), dim]
            ))
        )

        self.grid_fitness = torch.tensor(
            self.grid_fitness.scatter(0, alive_ind, torch.vectorized_map(
                fitness_fn, self.grid_values[alive_ind]
            ))
        )

        # self.grid_values[alive_ind] = torch.randint(
        #     domain[0], domain[1] + 1, [len(alive_ind), dim]
        # )
        self.grid_fitness[alive_ind] = torch.vectorized_map(
            fitness_fn, self.grid_values[alive_ind]
        )
        self.grid_alive[alive_ind] = True
        self.fract_broadcast = fract_broadcast
        self.fract_duplication = fract_duplication
        self.prob_die = prob_die

    def _broadcast_spawning(self, alive: torch.Tensor, n_broadcasters: int) -> torch.Tensor:
        broadcasters = self.grid_values[alive[:n_broadcasters]]

        crossover_points = torch.randint(
            0, broadcasters.shape[-1], [n_broadcasters // 2]
        )
        crossover_points = crossover_points.repeat(2)
        crossover_points = crossover_points.reshape(-1, 1)
        crossover_points = crossover_points.repeat(1, broadcasters.shape[1])
        broadcasters_swap = broadcasters.reshape(-1, 2, broadcasters.shape[-1])
        broadcasters_swap = torch.flip(broadcasters_swap, [1])
        broadcasters_swap = broadcasters_swap.reshape(-1, broadcasters.shape[-1])

        indexing = torch.arange(broadcasters.shape[1])
        indexing = indexing.reshape(1, -1)
        indexing = indexing.repeat(broadcasters.shape[0], 1)

        mask = indexing <= crossover_points

        new_corals = torch.where(mask, broadcasters, broadcasters_swap)
        return new_corals



    def _brooding(self, alive: torch.Tensor, n_broadcasters: int) -> torch.Tensor:
        brooders = self.grid_values[alive[n_broadcasters:]]

        if self.mutation_fn is not None:
            return self.mutation_fn(brooders)

        mutation = torch.randint(
            -self.mutation_range, self.mutation_range, brooders.shape
        )
        new_corals = brooders + mutation
        new_corals = torch.clip(new_corals, self.domain[0], self.domain[1])
        return new_corals
    
    def _larvae_settling(self, new_corals: torch.Tensor):
        for _ in range(self.settling_trials):
            if len(new_corals) == 0:
                break

            indices = torch.randperm(len(self.grid_values))[: len(new_corals)]

            alive = self.grid_alive[indices]
            old_fitness = self.grid_fitness[indices]
            new_fitness = torch.vectorized_map(self.fitness_fn, new_corals)
            settled_mask = ~alive | (new_fitness > old_fitness)
            settled_indices = indices[settled_mask]

            self.grid_values[settled_indices] = new_corals[settled_mask]
            self.grid_fitness[settled_indices] = new_fitness[settled_mask]
            self.grid_alive[settled_indices] = True

            new_corals = new_corals[~settled_mask]
    
    def _asexual_reproduction(self):
        n_duplication = int(self.fract_duplication * len(torch.where(self.grid_alive)))
        best_corals = self.grid_values[torch.topk(self.grid_fitness, n_duplication).indices]
        self._larvae_settling(best_corals)

    def _depredation(self):
        n_depradation = int(self.frac_duplication * len(torch.where(self.grid_alive)))
        coral_indices = torch.topk(-self.grid_fitness, n_depradation).indices
        coral_indices = coral_indices[torch.rand(coral_indices.shape[0]) < self.prob_die]
        self.grid_alive[coral_indices] = False

    def step(self):
        alive = torch.where(self.grid_alive)[0]
        torch.randperm(alive)

        n_broadcasters = int(self.fract_broadcast * len(alive)) // 2 * 2
        broadcasted = self._broadcast_spawning(alive, n_broadcasters)
        brooded = self._brooding(alive, n_broadcasters)

        settling_candidates = torch.cat([broadcasted, brooded], axis=0)
        self._larvae_settling(settling_candidates)

        self._asexual_reproduction()

        self._depredation()
    
    def best(self):
        best_coral = self.grid_values[torch.argmax(self.grid_fitness)]
        return best_coral, self.fitness_fn(best_coral)
    
def bit_flip(x, chance=0.1):
    return torch.where(torch.rand(x.shape) < chance, 1 - x, x)

def one_max(x):
    n = x.shape[0]
    return torch.sum(x) / n * 100


reef = CoralReef(
    fitness_fn=one_max,
    n_corals=10_000,
    domain=(0, 1),
    mutation_range=1,
    dim=7,
    dtype=torch.int32,
    settling_trials=10,
    mutation_fn=bit_flip,
)

for _ in range(10):
    reef.step()
    # alive = tf.where(reef.grid_alive).shape[0]
    # print(reef.best()[1].numpy(), tf.where(reef.grid_alive).shape[0])
    alive = torch.where(reef.grid_alive)[0].shape[0]
    print(reef.best()[1].numpy(), alive)


# bit_flip(tf.constant([[0, 1, 1, 0, 1]], dtype=tf.int32))