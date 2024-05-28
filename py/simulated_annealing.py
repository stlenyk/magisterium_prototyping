import torch
from typing import Callable
from common import bit_flip_prob, one_max


class SimulatedAnnealing:
    def __init__(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        domain: tuple[int, int],
        steps: int = 1000,
        mutation_fn: Callable[[torch.Tensor], torch.Tensor] = bit_flip_prob(p=0.5),
        device: torch.device = "cpu",
    ) -> None:
        self.fitnes_fn = fitness_fn
        self.dim = dim
        self.domain = domain
        self.mutation_fn = mutation_fn
        self.device = device
        self.steps = steps
        self.best = None

    def run(self):
        s = torch.randint(
            self.domain[0],
            self.domain[1] + 1,
            (self.dim,),
            device=self.device,
            dtype=torch.int32,
        )
        fit = self.fitnes_fn(s)
        self.best = s

        for k in range(self.steps):
            new_s = self.mutation_fn(s)
            new_fit = self.fitnes_fn(new_s)
            t = 1 - k / self.steps
            if self._acceptance_prob(fit, new_fit, t) > torch.rand(1, device=self.device):
                if new_fit < fit:
                    self.best = new_s
                s = new_s
                fit = new_fit

    def _acceptance_prob(self, e, new_e, t):
        if new_e < e:
            return 1.0
        else:
            return torch.exp((e - new_e) / t)


from timeit import default_timer as timer

t0 = timer()

algorithm = SimulatedAnnealing(
    fitness_fn=one_max,
    dim=500,
    domain=(0, 1),
    steps=100_000,
    mutation_fn=bit_flip_prob(p=0.5),
    device="cuda",
)

algorithm.run()
print(-one_max(algorithm.best))
elapsed_time = timer() - t0
print(f"Elapsed time: {elapsed_time:.2f}s")
