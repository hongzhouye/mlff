#!/bin/sh

for I in `seq 1 1 15`
do
	declare -i II=I+100
	cp data${I}.dat ../ding/data${II}.dat
done

for I in `seq 76 1 100`
do
	cp data${I}.dat ../ding/data${I}.dat
done
