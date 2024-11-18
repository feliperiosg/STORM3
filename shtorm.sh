#!/bin/bash

YEAR=($(seq 2024 1 2055))

for ii in "${YEAR[@]}"; do
	#echo "computing ${ii}"
	python storm.py -n 1 -y 1 -a ${ii}
done
