#!/bin/bash
set -e

list=$(ls)
for foo in ${list}
do
    if [[ ${foo##*.} == root ]]
    then
        if ! echo "${list[@]}" | grep -w "${foo%.*}.h5" &>/dev/null
        then
            echo ../point_axish5/${foo%.*}.h5
            /home/xdc/root2hdf5/C++/build/ConvertSimData ${foo} ../point_axish5/${foo%.*}.h5
        fi
    fi
done
