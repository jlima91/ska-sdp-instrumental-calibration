#!/usr/bin/env bash

# Script to run oskar simulations to generate SKA-LOW visibilities
# Authored by: Team Dhruva

set -e

## Parameters to be set by the user
SCENARIO=low40s-model

OSKAR_SIF="./OSKAR-2.11.1-Python3.sif"
WSCLEAN_CMD=${WSCLEAN_CMD:-"wsclean"}
DP3_CMD=${DP3_CMD:-"DP3"}

TEL_MODEL="SKA-Low_AA2_40S_rigid-rotation_model.tm"

START_FREQ_HZ="123e6"
END_FREQ_HZ="153e6"
CHANNEL_WIDTH_HZ="21.70138888888889e3"

OBS_LENGTH_MINS=10
DUMP_TIME_SEC="3.3973862400000003"

GLEAM_FILE="./sky-models/GLEAM_EGC.fits"
FIELD_RADIUS_DEG="10.0"

# GAINTABLE="gaintables/gain_model_40s_lg3.h5"
# CABLE_DELAY="cable_delays/cable_length_error_40s.txt"
# TEC_SCREEN="tec/calibrator_iono_tec.fits"

# Any extra parameters passed directly to run_oskar.py
# See the help for run_oskar.py: `singularity exec $OSKAR_SIF python3 run_oskar.py --help`
EXTRA_PARAMS="--use-gpus --double-precision" # "--max-sources-per-chunk 128"

# Image parameters
CREATE_DIRTY_IMAGE=1 # Comment this to not create image
IMAGE_SIZE=1024   # Reduce this to focus on the center source
PIXEL_SIZE=2arcsec

#######################################################
############# Don't change below this line ############
#######################################################

if [[ -n "$CREATE_DIRTY_IMAGE" ]] ; then
    # Early check for wsclean command
    if ! command -v "$WSCLEAN_CMD" &> /dev/null; then
        echo "wsclean command not found. Exiting" 1>&2;
        exit 1;
    fi

    # Early check for DP3 command
    if ! command -v "$DP3_CMD" &> /dev/null; then
        echo "DP3 command not found. Exiting" 1>&2;
        exit 1;
    fi
fi

WORKING_DIR=`builtin pwd`

## Create output directory in working dir
OUTPUT_DIR=$(realpath "$SCENARIO-`date +%d%m%y_%H%M%S`")
mkdir -p $OUTPUT_DIR

## Copy fields file
cp fields.yaml $OUTPUT_DIR/fields.yaml

## Symlink gleamfile if needed
if [[ -n "$GLEAM_FILE" ]]; then
    ln -s `realpath $GLEAM_FILE` $OUTPUT_DIR/GLEAM_EGC.fits
fi

## Create a temporary tm model
TEMP_TEL_MODEL="${OUTPUT_DIR}/`basename ${TEL_MODEL}`.custom"
cp -r $TEL_MODEL $TEMP_TEL_MODEL

## Add effects if user has provided a gain model
if [[ -n "$GAINTABLE" ]]; then
    ln -s `realpath $GAINTABLE` $TEMP_TEL_MODEL/gain_model.h5
fi

## Add cable delay errors if user has provided a error file
if [[ -n "$CABLE_DELAY" ]]; then
    ln -s `realpath $CABLE_DELAY` $TEMP_TEL_MODEL/cable_length_error.txt
fi

## Base singularity command
application="singularity exec \
-H $WORKING_DIR \
--nv \
`realpath $OSKAR_SIF`"

## Run options for the application:
options="python3 $WORKING_DIR/run_oskar.py \
--output-dir $OUTPUT_DIR \
--tel-model $TEMP_TEL_MODEL \
--obs-length-mins $OBS_LENGTH_MINS \
--dump-time-sec $DUMP_TIME_SEC \
--start-freq-hz $START_FREQ_HZ \
--end-freq-hz $END_FREQ_HZ \
--channel-width-hz $CHANNEL_WIDTH_HZ \
--field EoR2 \
--target Cal1 \
--scan-index 0 \
--num-scans 1"

## Add ionospheric tec screen if user has provided one
if [[ -n "$TEC_SCREEN" ]]; then
    options="$options --tec-screen `realpath $TEC_SCREEN`"
fi

## Add gleamfile option if file is provided
if [[ -n "$GLEAM_FILE" ]]; then
    options="$options --add-gleam --field-radius-deg $FIELD_RADIUS_DEG"
else
    options="$options --no-add-gleam"
fi

## Append user provided additional params
if [[ -n "$EXTRA_PARAMS" ]]; then
    options="$options $EXTRA_PARAMS"
fi

## Merge commands
CMD="$application $options"

## Start OSKAR simulation
cd $OUTPUT_DIR

echo "Time: `date`" &>> $OUTPUT_DIR/run_sim_py.out
echo "Current directory: `builtin pwd`" &>> $OUTPUT_DIR/run_sim_py.out

echo -e "\nExecuting command:\n==================\n$CMD\n" &>> $OUTPUT_DIR/run_sim_py.out

eval $CMD &>> $OUTPUT_DIR/run_sim_py.out

## Generate image using wsclean

if [[ -n "$CREATE_DIRTY_IMAGE" ]] ; then
    SIM_MS=`find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*.ms" -print -quit`
    BEAM_CORRECTED_MS=${SIM_MS}.beamcor.ms
    IMAGE_NAME="$OUTPUT_DIR/$SCENARIO-wsclean"
    export OPENBLAS_NUM_THREADS=1

    # Apply beam using DP3
    $DP3_CMD msin=$SIM_MS steps=[applybeam] msout=$BEAM_CORRECTED_MS

    # Then create image using wsclean on DP3's output
    $WSCLEAN_CMD -size $IMAGE_SIZE $IMAGE_SIZE \
    -scale $PIXEL_SIZE -niter 0 -apply-primary-beam \
    -name $IMAGE_NAME $BEAM_CORRECTED_MS

    # Remove DP3's output
    rm -rf $BEAM_CORRECTED_MS
fi