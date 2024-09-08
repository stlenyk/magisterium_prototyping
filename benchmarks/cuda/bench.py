import torch
import csv


def profile(func, out_name: str):
    with torch.profiler.profile() as prof:
        func()

    bench_res = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=50,
        max_name_column_width=21370,
    )

    lines = bench_res.split("\n")[3:-4]
    header = [
        "Name",
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
        "Self CUDA",
        "Self CUDA %",
        "CUDA total",
        "CUDA time avg",
        "# of Calls",
    ]
    res = [header]
    col_size = 14
    col_count = 10
    for line in lines:
        res.append([])
        old_shift = 0
        shift = col_size * col_count + 1
        for _ in range(col_count + 1):
            res[-1].append(line[-old_shift:-shift].strip())
            old_shift = shift
            shift -= col_size

    with open(f"{out_name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(res)
    
    with open(f"{out_name}.txt", "w") as f:
        f.write(bench_res)
    


from optimization_algorithms import func

def bench():
    n = 100_000_000
    x = torch.rand(dtype=torch.float32, device="cuda", size=(n,))
    y = torch.rand(dtype=torch.float32, device="cuda", size=(n,))
    for _ in range(1000):
        z = func.uniform_crossover(x, y)
    torch.cuda.synchronize()


profile(bench, "rand")