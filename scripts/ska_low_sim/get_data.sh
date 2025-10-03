#!/usr/bin/env bash

# Script to get necessary data needed to simulate SKA-LOW visibilities
# User needs to have access to the SKAO S3 buckets
# Authored by: Team Dhruva

DATA_PREFIX=s3://skao-sdp-testdata/PI27-Low-G4

TEL_MODEL_DIR=telescope-models
TEL_MODEL=SKA-Low_AA2_40S_rigid-rotation_model.tm

SKY_MODEL_DIR=sky-models
SKY_MODEL=GLEAM_EGC.fits

TEC_DIR=tec
TEC_FILE=scan_0.fits

# Get OSKAR singularity container
wget https://gitlab.com/ska-telescope/sim/oskar/-/jobs/10486879708/artifacts/raw/OSKAR-2.11.1-Python3.sif

# Get telescope model
aws s3 sync $DATA_PREFIX/$TEL_MODEL_DIR/$TEL_MODEL/ ./$TEL_MODEL_DIR/$TEL_MODEL/
# Move cable delays out of telmodel
mkdir cable_delays
mv ./$TEL_MODEL_DIR/$TEL_MODEL/cable_length_error.txt ./cable_delays/cable_length_error_40s.txt

# Get gleamfile
aws s3 sync $DATA_PREFIX/$SKY_MODEL_DIR/$SKY_MODEL ./$SKY_MODEL_DIR/$SKY_MODEL

# Get ionospheric screen
aws s3 sync $DATA_PREFIX/$TEC_DIR/$TEC_FILE ./$TEC_DIR/$TEC_FILE
