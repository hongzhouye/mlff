#!/bin/sh

for I in `seq 1 1 25`;
do
	declare -i K=I+25
	declare -i L=I+15
	mv data${K}.dat data${L}.dat
done
