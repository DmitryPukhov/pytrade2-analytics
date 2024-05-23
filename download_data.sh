#!/bin/bash
# Download new candles to data folder
days=${1:-2}

PROJECT_DIR=$(pwd)
PYTRADE2_DIR=$PROJECT_DIR/pytrade2
DATA_DIR=$PROJECT_DIR/data

echo "Downloading new candles for days:$days  to $DATA_DIR"
today=$(date '+%Y-%m-%d')

cd $PYTRADE2_DIR/pytrade2
python DataDownloadApp.py --pytrade2.data.dir $DATA_DIR --to $today --days $days
