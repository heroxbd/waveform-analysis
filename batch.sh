#!/bin/sh
set -e

rawds=$(ls /mnt/eternity/Jinping_1ton_Data/01_RawData)

for rawd in ${rawds}
do
    date
    make iptfold=/mnt/eternity/Jinping_1ton_Data/01_RawData/${rawd} optfold=/mnt/eternity/Jinping_1ton_Data/Charge/${rawd} method=lucyddm
    date
done
