#!/bin/bash
#SBATCH -A m4265
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -t 02:13:20
#SBATCH -N 64

export EXESS_PATH="/global/homes/r/ryans/HERMES"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation
INPUT_DIR=$(pwd)

OUT_DIR="../../outputs/exess/n_2beg_64_nodes"
mkdir -p $OUT_DIR
pushd $OUT_DIR
rm *_logs/*
srun -N 64 --export=ALL --ntasks-per-node=9 --gpus-per-node=4 $EXESS_PATH/build/exess $INPUT_DIR/n_2beg_64_nodes.json 2>&1 | tee n_2beg_64_nodes.out
popd
                         