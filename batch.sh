#!/bin/sh
set -e

rawds=$(ls /mnt/eternity/Jinping_1ton_Data/01_RawData)

for rawd in ${rawds}
do
    make datfold=/mnt/eternity/Jinping_1ton_Data/01_RawData/${rawd} method=lucyddm -j150
done
