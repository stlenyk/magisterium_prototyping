from optimization_algorithms import func

import torch


class SimulatedAnnealing:
    def __init__(
        self,
        fitness_fn: func.TensorFn,
        dim: int,
        domain: func.Domain,
        dtype: torch.dtype = None,
        n_steps: int = 1000,
        mutation_fn: func.TensorFn = None,
        device: torch.device = "cpu",
    ) -> None:
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.domain = domain
        self.dtype = dtype
        self.n_steps = n_steps
        self.mutation_fn = mutation_fn
        self.device = device
        self._best = None
        self._best_fit = None

    def run(self):
        s = torch.randint(
            self.domain[0],
            self.domain[1] + 1,
            (self.dim,),
            device=self.device,
            dtype=self.dtype,
        )
        fit = self.fitness_fn(s)
        self._best = s
        self._best_fit = fit

        for k in range(self.n_steps):
            new_s = self.mutation_fn(s)
            new_fit = self.fitness_fn(new_s)
            t = 1.0 - (k / self.n_steps)
            if self._acceptance_prob(fit, new_fit, t) > torch.rand(
                1, device=self.device
            ):
                if new_fit < self._best_fit:
                    self._best = new_s
                    self._best_fit = new_fit
                s = new_s
                fit = new_fit

    @staticmethod
    def _acceptance_prob(e, new_e, t):
        if new_e < e:
            return 1.0
        else:
            return torch.exp((e - new_e) / t)

    def best(self):
        return self._best, self._best_fit
