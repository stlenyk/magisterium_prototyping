import torch


def mutation_n(domain_clip, draw_range, n=1):
    def wrapper(x):
        indices = torch.randint(x.shape[0] + 1, (n,), device=x.device)
        values = torch.randint(
            draw_range[0], draw_range[1] + 1, (n,), device=x.device, dtype=x.dtype
        )
        values = torch.clip(values, domain_clip[0], domain_clip[1])
        x[indices] += values
        return x

    return wrapper


def mutation_prob(domain_clip, draw_range, probability=0.1):
    def wrapper(x: torch.Tensor):
        mask = torch.rand(x.shape, device=x.device) < probability
        values = torch.randint(
            draw_range[0], draw_range[1] + 1, x.shape, device=x.device, dtype=x.dtype
        )
        values = torch.clip(values, domain_clip[0], domain_clip[1])
        return torch.where(mask, x + values, x)

    return wrapper


def bit_flip_n(n=1):
    def wrapper(x: torch.Tensor):
        indices = torch.randint(x.shape[0] + 1, (n,), device=x.device)
        x[indices] = 1 - x[indices]
        return x

    return wrapper


def bit_flip_prob(p=0.1):
    def wrapper(x: torch.Tensor):
        return torch.where(torch.rand(x.shape, device=x.device) < p, 1 - x, x)

    return wrapper


def uniform_crossover(x: torch.Tensor, y: torch.Tensor):
    mask = torch.rand(x.shape, device=x.device) < 0.5
    return torch.where(mask, x, y)


def single_point_crossover(x: torch.Tensor, y: torch.Tensor):
    crossover_point = torch.randint(0, x.shape[-1], [1], device=x.device)
    mask = torch.arange(x.shape[-1], device=x.device) < crossover_point
    return torch.where(mask, x, y)


def one_max(x: torch.Tensor):
    n = x.shape[0]
    return -torch.sum(x) / n * 100
