#!/bin/sh

for I in `seq 26 1 50`;
do
	declare -i K=I-25
	mv data${I}.dat data${K}.dat
done
