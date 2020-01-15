#!/bin/bash

rm -f log.txt
date >> log.txt

for i in $(seq 0 99)
do
echo ${i} >> log.txt
python3 -m memory_profiler test/xiaopeip/memopfinal.py -c ${i} | grep 49 >> log.txt
date >> log.txt
done
