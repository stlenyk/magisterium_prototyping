#!/bin/bash

# This script is used to run the benchmark for the given number of iterations

ITERS=10
for ((i = 1; i <= $ITERS; i++)); do
    echo "$i"/"$ITERS"
    time python -m benchmarks.vmap.bench_vmap  
done
