#!/bin/bash

# N_SMALL="25,50,100,200,400,800,1600,3200,6400"
# N_LARGE="25,50,100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800"

# make N_SMALL and N_LARGE smallery by a factor of 10
N_SMALL="2,5,10,20,40,80,160,320,640"
N_LARGE="2,5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480"

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

hyperfine "python $ALG_DIR/my_sa.py cuda 500 {n_population}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_sa_cuda.csv" \
    --parameter-list n_population $N_LARGE

hyperfine "python $ALG_DIR/my_sa.py cpu {n_dim}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"my_sa_cpu.csv" \
    --parameter-list n_dim $N_SMALL

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
    --parameter-list n_population $N_SMALL

hyperfine "python $ALG_DIR/their_sa.py {n_dim}" \
    --warmup $HYP_N_WARMUP \
    --runs $HYP_N_RUNS \
    --export-csv $RES_DIR/"their_sa.csv" \
    --parameter-list n_dim $N_SMALL
