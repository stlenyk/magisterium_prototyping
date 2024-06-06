import torch
from timeit import default_timer as timer

from src.common import one_max, single_point_crossover

ITERS = 1000
DIM = 1000
BATCH = 100_000


t = torch.randint(0, 2, (BATCH, DIM), dtype=torch.int32, device="cuda")


def time_fn(fn, *args, out_file=None):
    res = fn(*args)  # warmup
    torch.cuda.synchronize()
    start = timer()
    for _ in range(ITERS):
        res = fn(*args)
    torch.cuda.synchronize()
    end = timer()
    elapsed = end - start
    print(f"{elapsed:.2f} s")

    if out_file:
        with open(f"{out_file}", "a") as f:
            f.write(f"{elapsed}\n")


def one_max_batch(t):
    return torch.sum(t, dim=1) / t.shape[1] * 100


def single_point_crossover_batch(t):
    t_swap = t.reshape(t.shape[0] // 2, 2, t.shape[1])
    t_swap = torch.flip(t_swap, [1])
    t_swap = t_swap.reshape(-1, t.shape[-1])

    crossover_points = torch.randint(
        0, t.shape[-1] + 1, (t.shape[0] // 2,), device=t.device
    )
    crossover_points = crossover_points.repeat_interleave(2)
    crossover_points = crossover_points.reshape(-1, 1)
    crossover_points = crossover_points.repeat(1, t.shape[1])

    indexing = torch.arange(t.shape[1], device=t.device)
    indexing = indexing.reshape(1, -1)
    indexing = indexing.repeat(t.shape[0], 1)

    mask = indexing < crossover_points

    new_t = torch.where(mask, t, t_swap)
    return new_t


time_fn(torch.vmap(one_max), t, out_file=f"bench_vmap_one_max.txt")
time_fn(one_max_batch, t, out_file=f"bench_vmap_one_max_batch.txt")
time_fn(
    torch.vmap(single_point_crossover, randomness="different"),
    t[::2],
    t[1::2],
    out_file="bench_vmap_crossover.txt",
)
time_fn(single_point_crossover_batch, t, out_file="bench_vmap_crossover_batch.txt")
