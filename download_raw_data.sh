#!/bin/bash
# Download new candles to data folder

PROJECT_DIR=$(pwd)
S3_DIR=s3://pytrade2/data/raw
LOCAL_DIR=$PROJECT_DIR/data

echo "Downloading new data from $S3_DIR to $LOCAL_DIR"
s3cmd -v sync $S3_DIR $LOCAL_DIR

