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
        indices = torch.randint(x.shape[0], (n,), device=x.device)
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


def _bit3_deceptive_single_group(x: torch.Tensor) -> torch.Tensor:
    """Calculates the 3-bit deceptive problem for a single 3-bit group.

    3-bit group | Value
    -------------------
    111         | 80
    000         | 70
    001         | 50
    010         | 49
    100         | 30
    110         | 3
    101         | 2
    011         | 1
    """

    device = x.device
    return (
        torch.tensor([1, 1, 1], device=device).eq(x).all() * 80
        + torch.tensor([0, 0, 0], device=device).eq(x).all() * 70
        + torch.tensor([0, 0, 1], device=device).eq(x).all() * 50
        + torch.tensor([0, 1, 0], device=device).eq(x).all() * 49
        + torch.tensor([1, 0, 0], device=device).eq(x).all() * 30
        + torch.tensor([1, 1, 0], device=device).eq(x).all() * 3
        + torch.tensor([1, 0, 1], device=device).eq(x).all() * 2
        + torch.tensor([0, 1, 1], device=device).eq(x).all() * 1
    )


def bit3_deceptive(x: torch.Tensor) -> torch.Tensor:
    """Fitness function. Calculates the 3-bit deceptive problem.

    3-bit group | Value
    -------------------
    111         | 80
    000         | 70
    001         | 50
    010         | 49
    100         | 30
    110         | 3
    101         | 2
    011         | 1
    """

    return torch.vmap(_bit3_deceptive_single_group)(x.reshape(-1, 3)).sum()


def travelling_salesman(
    d: torch.Tensor,
) -> TensorFn:
    """Fitness function. Calculates the total distance of a path in the Travelling Salesman Problem.

    d - distance matrix, where d[i, j] is the distance between cities i and j.
    Produces a function that takes path, i.e. a permutation of [0, 1, ..., n-1], and returns the total distance.
    """
    inf = d.max() * d.shape[0]

    @functools.wraps(travelling_salesman)
    def wrapper(x: torch.Tensor) -> torch.Tensor:
        wrong_condition = (
            ~torch.eq(
                torch.sort(x).values, torch.arange(x.shape[0], device=x.device)
            ).all()
            * inf
        )
        tsp = torch.sum(d[x, torch.roll(x, 1, dims=0)])
        return (wrong_condition + tsp).to(torch.float32)

    return wrapper
