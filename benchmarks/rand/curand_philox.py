import torch
from timeit import default_timer as timer


# torch uses cuRAND's Philox algorithm for generating random numbers on CUDA

n = 100_000_000
iters = 1000
device = "cuda"

x = torch.rand(n, device=device, dtype=torch.float32)
torch.cuda.synchronize()
t0 = timer()
for _ in range(iters):
    x = torch.rand(n, device=device, dtype=torch.float32)
torch.cuda.synchronize()
elapsed = timer() - t0
print(f"GSamples/sec", "{:.1f}".format(iters * n / elapsed / 1e9))
