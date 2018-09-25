#!/bin/bash

for i in {1..10}
do
	echo "Run ${i}"
	./threaded-knn datasets/medium.arff
done