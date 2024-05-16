import torch


def bit_flip_chance(chance=0.1):
    def wrapper(x):
        return torch.where(torch.rand(x.shape, device=x.device) < chance, 1 - x, x)

    return wrapper


def bit_flip_n(n=1):
    def wrapper(x):
        indices = torch.randint(0, x.shape[0], (n,), device=x.device)
        x[indices] = 1 - x[indices]
        return x

    return wrapper


def uniform_crossover(x, y):
    mask = torch.rand(x.shape, device=x.device) < 0.5
    return torch.where(mask, x, y)


def single_point_crossover(x, y):
    crossover_point = torch.randint(0, x.shape[-1], [1], device=x.device)
    mask = torch.arange(x.shape[-1], device=x.device) < crossover_point
    return torch.where(mask, x, y)


def one_max(x):
    n = x.shape[0]
    return -torch.sum(x) / n * 100
