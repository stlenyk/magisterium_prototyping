import torch
import matplotlib.pyplot as plt


def generate_insert_tensor(n_points: int, n_duplicates: int, device) -> torch.Tensor:
    indices = torch.arange(n_points, device=device).repeat(n_duplicates)
    indices = indices[torch.randperm(indices.shape[0], device=device)]
    points = torch.zeros(n_points * n_duplicates, device=device)
    counters = torch.zeros(n_points, device=device)
    for i, idx in enumerate(indices):
        counters[idx] += 1
        points[i] = counters[idx]
    return indices, points


def generate_1_dup(n_points: int, device) -> torch.Tensor:
    dup_idx = torch.randint(0, n_points, (1,), device=device)
    indices = torch.cat((torch.arange(n_points, device=device), dup_idx))
    indices = indices[torch.randperm(indices.shape[0], device=device)]
    points = torch.ones(n_points + 1, device=device)
    dup_idx_idx = torch.where(indices == dup_idx)
    dup_idx_idx = dup_idx_idx[0][1]
    points[dup_idx_idx] = 2
    return indices, points


# total_points = 1_000_000
# n_duplicates = 2
# n_points = total_points // n_duplicates
# indices, points = generate_insert_tensor(n_points, n_duplicates)
# t = torch.zeros(n_points, device=device)
# t[indices] = points
# good_res = torch.sum(torch.where(t == n_duplicates, 1, 0))
# print("mean idx", torch.mean(t))
# print("percent good res", good_res / t.shape[0])

iters = 100
bench_cases = [100]
while bench_cases[-1] <= 10_000_000:
    bench_cases.append(int(bench_cases[-1] * 1.37))
res = [[], []]
for device_idx, device in enumerate(["cpu", "cuda"]):
    for n_points in bench_cases:
        res_i = 0
        for _ in range(iters):
            indices, points = generate_1_dup(n_points, device)
            t = torch.zeros(n_points, device=device)
            t[indices] = points
            good_res = len(torch.where(t == 2)[0])
            res_i += good_res
        print(device, n_points, res_i / iters)
        res[device_idx].append(res_i / iters)

with open("results.csv", "w") as f:
    f.write("tensor_size,cpu,cuda\n")
    for i in range(len(bench_cases)):
        f.write(f"{bench_cases[i]},{res[0][i]},{res[1][i]}\n")
