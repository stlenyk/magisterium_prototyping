import torch


class OptimizationAlgorithm:
    def __init__(self, fitness_fn, dim, domain, device):
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.domain = domain
        self.device = device


def bit_flip_chance(x, device, chance=0.1):
    return torch.where(torch.rand(x.shape, device=device) < chance, 1 - x, x)


def bit_flip_n(device, n=1):
    def wrapped(x):
        indices = torch.randint(0, x.shape[0], (n,), device=device)
        x[indices] = 1 - x[indices]
        return x

    return wrapped


def one_max(x):
    n = x.shape[0]
    return -torch.sum(x) / n * 100
