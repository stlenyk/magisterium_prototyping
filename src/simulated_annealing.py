import common

import torch


class SimulatedAnnealing:
    def __init__(
        self,
        fitness_fn: common.TensorFn,
        dim: int,
        domain: common.Domain,
        dtype: torch.dtype = None,
        steps: int = 1000,
        mutation_fn: common.TensorFn = None,
        device: torch.device = "cpu",
        logging: bool = False,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.domain = domain
        self.dtype = dtype
        self.steps = steps
        self.mutation_fn = mutation_fn
        self.device = device
        self.logging = logging
        self.best = None
        self.best_fit = None

    def run(self):
        s = torch.randint(
            self.domain[0],
            self.domain[1] + 1,
            (self.dim,),
            device=self.device,
            dtype=self.dtype,
        )
        fit = self.fitness_fn(s)
        self.best = s
        self.best_fit = fit

        for k in range(self.steps):
            # log every 1% of the steps
            if self.logging and k % (self.steps // 100) == 0:
                print(f"{k / self.steps * 100:.0f}%", -self.fitness_fn(self.best))
            new_s = self.mutation_fn(s)
            new_fit = self.fitness_fn(new_s)
            t = 1.0 - (k / self.steps)
            if self._acceptance_prob(fit, new_fit, t) > torch.rand(
                1, device=self.device
            ):
                if new_fit < self.best_fit:
                    self.best = new_s
                    self.best_fit = new_fit
                s = new_s
                fit = new_fit

    @staticmethod
    def _acceptance_prob(e, new_e, t):
        if new_e < e:
            return 1.0
        else:
            return torch.exp((e - new_e) / t)


if __name__ == "__main__":
    from timeit import default_timer as timer

    t0 = timer()

    algorithm = SimulatedAnnealing(
        fitness_fn=common.one_max,
        dim=50_000_000,
        domain=(0, 1),
        steps=1_000,
        mutation_fn=common.bit_flip_prob(p=0.5),
        device="cuda",
        logging=True,
    )

    algorithm.run()
    print(-algorithm.best_fit)
    elapsed_time = timer() - t0
    print(f"Elapsed time: {elapsed_time:.2f}s")
