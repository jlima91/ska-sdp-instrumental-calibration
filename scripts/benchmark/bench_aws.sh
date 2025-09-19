#!/usr/bin/env bash

set -e

# ** Top-level parameters (to be supplied by team)
BENCH_NAME=inst
# Repository to use
REPOSITORY=https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration.git
BRANCH=main
BENCH_FOLDER=scripts/benchmark
# Input, from S3
INPUT_S3="s3://skao-sdp-testdata/bench/inputs/$BENCH_NAME"
# Partition and node count
PARTITION=hpc7a-48xl-ond
NODE_COUNT=3

# ** Platform parameters (set per platform)
BENCHID=$BENCH_NAME-`date +%Y-%m-%d`
# Might instead just-in-time compile at some point
export MODULEPATH=/shared/fsx1/shared/metamodules
export META_MODULE=ska-sdp-spack/2025.08.3

# ** Preparation

# Check out repository and dataset
export BENCH_PATH=/shared/fsx1/bench/$BENCHID
export INPUT_PATH=$BENCH_PATH/input
export OUTPUT_PATH=$BENCH_PATH/output
export REPORT_PATH=$BENCH_PATH/report
export CODE_PATH=$BENCH_PATH/code
LOG_PATH=$BENCH_PATH/logs
mkdir -p $INPUT_PATH
mkdir -p $OUTPUT_PATH
mkdir -p $REPORT_PATH
mkdir -p $CODE_PATH
mkdir -p $LOG_PATH

cd $BENCH_PATH

# Do sparse-checkout for the specific script
cd $CODE_PATH
git init -b $BRANCH
git remote add origin $REPOSITORY
git sparse-checkout init
git sparse-checkout set $BENCH_FOLDER
git pull --set-upstream origin $BRANCH
cd -

# Copy input data from S3
aws configure set default.s3.max_concurrent_requests 100
aws s3 cp --recursive $INPUT_S3 $INPUT_PATH

# Run pipeline
cd $LOG_PATH
sbatch --wait -p $PARTITION --nodes=$NODE_COUNT $CODE_PATH/$BENCH_FOLDER/inst.sh

# Remove inputs
rm -rf $INPUT_PATH

# Upload results to S3
aws s3 cp $BENCH_PATH s3://ska-sdp-testdata/bench/$BENCHID