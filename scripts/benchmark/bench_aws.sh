#!/bin/bash

set -e

# ** Top-level parameters (to be supplied by team)
BENCH_NAME=inst
# Repository to use
REPOSITORY=https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration
# Script to run. Should include most of the non-platform-specific parameters
SCRIPT=scripts/benchmark/inst.sh
# Input, from S3
INPUT_S3_BUCKET=skao-sdp-testdata
INPUT_S3_PREFIX=dhruva
INPUT_S3_PATH=inst-benchmark-input
INPUT_CACHE_PATH=/shared/fsx1/shared/product
# Partition and node countß
PARTITION=hpc7a-48xl-ond
NODE_COUNT=1

# ** Platform parameters (set per platform)
BENCHID=$BENCH_NAME-`date +%Y-%m-%d`
# Might instead just-in-time compile at some point
MODULEPATH=/shared/fsx1/shared/metamodules
META_MODULE=ska-sdp-spack/2025.08.3

# ** Preparation
BENCH_PATH=/shared/fsx1/bench/$BENCHID
if [ -e $BENCH_PATH ]; then
    for ((i=1;;i++)); do
        if [ ! -e $BENCH_PATH-$i ]; then break; fi
    done
    BENCH_PATH=$BENCH_PATH-$i
fi
echo Benchmark path: $BENCH_PATH
INPUT_PATH=$BENCH_PATH/input
OUTPUT_PATH=$BENCH_PATH/output
REPORT_PATH=$BENCH_PATH/report
CODE_PATH=$BENCH_PATH/code
LOG_PATH=$BENCH_PATH/logs
mkdir -p $OUTPUT_PATH
mkdir -p $REPORT_PATH
mkdir -p $LOG_PATH

# Synchronise input data from S3
cd $LOG_PATH
aws configure set default.s3.max_concurrent_requests 100
if [ ! -e $INPUT_CACHE_PATH/$INPUT_S3_PATH ]; then
    echo $INPUT_CACHE_PATH/$INPUT_S3_PATH does not exist, creating...
    aws s3 sync s3://$INPUT_S3_BUCKET/$INPUT_S3_PREFIX/$INPUT_S3_PATH $INPUT_CACHE_PATH/$INPUT_S3_PATH > s3sync.log
else
    echo s3://$INPUT_S3_BUCKET/$INPUT_S3_PREFIX/$INPUT_S3_PATH
    aws s3 sync --dryrun s3://$INPUT_S3_BUCKET/$INPUT_S3_PREFIX/$INPUT_S3_PATH $INPUT_CACHE_PATH/$INPUT_S3_PATH > s3sync-dryrun.log
    if [ -s s3sync-dryrun.log ]; then
        echo $INPUT_CACHE_PATH/$INPUT_S3_PATH is different, creating a copy...
        aws s3 sync s3://$INPUT_S3_BUCKET/$INPUT_S3_PREFIX/$INPUT_S3_PATH $INPUT_CACHE_PATH/$INPUT_S3_PATH
    else
        echo $INPUT_CACHE_PATH/$INPUT_S3_PATH is unchanged, will use it...
    fi
fi
ln -s $INPUT_CACHE_PATH/$INPUT_S3_PATH $INPUT_PATH

# inst.sh specific variables
export PRE_PROCESSED_CALIBRATOR=$INPUT_PATH/pre-processed-calibrator-68s-rigid-rotation.ms
export CALIBRATOR_SKY_MODEL=$INPUT_PATH/sky_model.csv

# Check out repository
git clone $REPOSITORY $CODE_PATH

# Run pipeline. We set HOME to BENCH_PATH, a couple of Python libraries use it as cache.
env -i MODULEPATH=$MODULEPATH META_MODULE=$META_MODULE INPUT_PATH=$INPUT_PATH \
       OUTPUT_PATH=$OUTPUT_PATH REPORT_PATH=$REPORT_PATH CODE_PATH=$CODE_PATH \
       HOME=$BENCH_PATH \
  /bin/bash -c ". /etc/profile && sbatch --wait -p $PARTITION --nodes=$NODE_COUNT $CODE_PATH/$SCRIPT"

# Upload results to S3
aws s3 cp $BENCH_PATH s3://$INPUT_S3_BUCKET/bench/$BENCHID