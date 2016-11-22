#!/bin/sh

for I in `seq 1 1 25`;
do
	mv data${I}.dat random
done

for I in `seq 26 1 50`;
do
	mv data${I}.dat md
done
