#!/bin/bash

N_SMALL="10,20,50,100,200,400,800,1600,3200,6400"
N_LARGE="10,20,50,100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800"

HYP_N_WARMUP=3
HYP_N_RUNS=10

RES_DIR="results"
ALG_DIR="alg_scripts"

###############################################################################
# mine                                                                        #
###############################################################################

hyperfine "python $ALG_DIR/my_cro.py cuda 500 {n_population}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_cro_cuda.csv" \
    --parameter-list n_population $N_LARGE

hyperfine "python $ALG_DIR/my_cro.py cpu 500 {n_population}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_cro_cpu.csv" \
    --parameter-list n_population $N_SMALL

hyperfine "python $ALG_DIR/my_abc.py cuda 500 {n_population}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_abc_cuda.csv" \
    --parameter-list n_population $N_LARGE

hyperfine "python $ALG_DIR/my_abc.py cpu 500 {n_population}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_abc_cpu.csv" \
    --parameter-list n_population $N_SMALL

hyperfine "python $ALG_DIR/my_sa.py cuda {n_dim}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_sa_cuda.csv" \
    --parameter-list n_dim "10,20,50,100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800,51200,102400,204800,409600,819200,1638400,3276800,6553600,13107200,26214400,52428800,104857600"

hyperfine "python $ALG_DIR/my_sa.py cpu {n_dim}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_sa_cpu.csv" \
    --parameter-list n_dim "10,20,50,100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800,102400,204800,409600,819200,1638400,3276800,6553600"

###############################################################################
# theirs                                                                      #
###############################################################################

hyperfine "python $ALG_DIR/their_cro.py 500 {n_population}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"their_cro.csv" \
    --parameter-list n_population $N_SMALL

hyperfine "python $ALG_DIR/their_abc.py 500 {n_population}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"their_abc.csv" \
    --parameter-list n_population "10,20,50,100,200,400"

hyperfine "python $ALG_DIR/their_sa.py {n_dim}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"their_sa.csv" \
    --parameter-list n_dim "10,20,50,100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800"
