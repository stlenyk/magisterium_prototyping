from timeit import default_timer as timer

from optimization_algorithms import func
import torch


def time_function(func, *args, iters=10, warmup=3):
    torch.cuda.synchronize()
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start = timer()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    end = timer()

    return (end - start) / iters


n = 100_000_000
iters = 10
x = torch.rand((n,), device="cpu")
res_cpu = time_function(func.one_max, x, iters=iters)
print("OneMax CPU", f"{res_cpu:.1e}s")
x = torch.rand((n,), device="cuda")
res_gpu = time_function(func.one_max, x, iters=iters)
print("OneMax GPU", f"{res_gpu:.1e}s")
print("Speedup", f"{res_cpu / res_gpu:.1f}")


n = 150_000_000
iters = 10
x = torch.rand((n,), device="cpu")
res_cpu = time_function(func.bit3_deceptive, x, iters=iters)
print("Bit3Deceptive CPU", f"{res_cpu:.1e}s")
x = torch.rand((n,), device="cuda")
res_gpu = time_function(func.bit3_deceptive, x, iters=iters)
print("Bit3Deceptive GPU", f"{res_gpu:.1e}s")
print("Speedup", f"{res_cpu / res_gpu:.1f}")


n = 20_000
iters = 10
d = torch.rand((n, n), device="cpu")
x = torch.randperm(n, device="cpu")
res_cpu = time_function(func.travelling_salesman(d), x, iters=iters)
print("TSP CPU", f"{res_cpu:.1e}s")
d = torch.rand((n, n), device="cuda")
x = torch.randperm(n, device="cuda")
res_gpu = time_function(func.travelling_salesman(d), x, iters=iters)
print("TSP GPU", f"{res_gpu:.1e}s")
print("Speedup", f"{res_cpu / res_gpu:.1f}")
