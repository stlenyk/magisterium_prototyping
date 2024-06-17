import functools
from typing import TypeAlias, Callable

import torch


DType: TypeAlias = int | float | bool
Domain: TypeAlias = tuple[int, int] | tuple[float, float] | tuple[bool, bool]
TensorFn: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


def rand_range_infer(
    input: torch.Tensor, size: tuple[int], range: Domain
) -> torch.Tensor:
    """Returns a tensor filled with random values. Infers the dtype and device from input.

    Args:
        input: The input tensor to infer the dtype and device from.
        size: The size of the output tensor.
        domain: The domain to draw the random values from.

    Raises:
        ValueError: If the `input` tensor is complex.
    """
    if torch.is_complex(input):
        raise ValueError("No support for complex tensors.")

    if torch.is_floating_point(input):
        return (
            torch.rand(size, dtype=input.dtype, device=input.device)
            * (range[1] - range[0])
            + range[0]
        )
    else:
        return torch.randint(
            range[0], range[1] + 1, size, device=input.device, dtype=input.dtype
        )


def mutation_n(domain_clip: Domain, draw_range: Domain, n: int = 1) -> TensorFn:
    """Returns a mutation function that adds n random values to n random indices of a tensor.

    Args:
        domain_clip: The domain to clip the random values to.
        draw_range: The range to draw the random values from.
        n: The number of random values to add.
    """

    @functools.wraps(mutation_n)
    def wrapper(x: torch.Tensor) -> torch.Tensor:
        indices = torch.randint(x.shape[0] + 1, (n,), device=x.device)
        values = torch.randint(
            draw_range[0], draw_range[1] + 1, (n,), device=x.device, dtype=x.dtype
        )
        values = torch.where(torch.rand(n, device=x.device) < 0.5, values, -values)
        x[indices] = torch.clip(x[indices] + values, domain_clip[0], domain_clip[1])
        return x

    return wrapper


def mutation_prob(
    domain_clip: Domain, draw_range: Domain, probability: float = 0.1
) -> TensorFn:
    """Returns a mutation function that adds random values to random indices of a tensor with a given probability.

    Args:
        domain_clip: The domain to clip the random values to.
        draw_range: The range to draw the random values from.
        probability: The probability [0.0, 1.0] of adding a random value to an index.
    """

    @functools.wraps(mutation_prob)
    def wrapper(x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) < probability
        values = torch.randint(
            draw_range[0], draw_range[1] + 1, x.shape, device=x.device, dtype=x.dtype
        )
        values = torch.where(
            torch.rand(x.shape, device=x.device) < probability, values, -values
        )
        mutated = torch.clip(x + values, domain_clip[0], domain_clip[1])
        return torch.where(mask, mutated, x)

    return wrapper


def bit_flip_n(n: int = 1) -> TensorFn:
    """Returns a mutation function that flips n random bits of a tensor.

    Args:
        n: The number of bits to flip.
    """

    @functools.wraps(bit_flip_n)
    def wrapper(x: torch.Tensor) -> torch.Tensor:
        indices = torch.randint(x.shape[0], (n,), device=x.device)
        x[indices] = 1 - x[indices]
        return x

    return wrapper


def bit_flip_prob(p: float = 0.1) -> TensorFn:
    """Returns a mutation function that flips bits of a tensor with a given probability.
    Args:
        p: The probability [0.0, 1.0] of flipping a bit.
    """

    @functools.wraps(bit_flip_prob)
    def wrapper(x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.rand(x.shape, device=x.device) < p, 1 - x, x)

    return wrapper


def uniform_crossover(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Performs uniform crossover between two tensors. Returns a single offspring tensor."""
    mask = torch.rand(x.shape, device=x.device) < 0.5
    return torch.where(mask, x, y)


def single_point_crossover(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Performs single-point crossover between two tensors. Returns a single offspring tensor."""
    crossover_point = torch.randint(0, x.shape[-1], [1], device=x.device)
    mask = torch.arange(x.shape[-1], device=x.device) < crossover_point
    return torch.where(mask, x, y)


def one_max(x: torch.Tensor) -> torch.Tensor:
    """Fitness function. Returns the percentage of ones in a boolean tensor."""
    n = x.shape[0]
    return -torch.sum(x) / n * 100
