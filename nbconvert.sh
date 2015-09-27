#!/usr/bin/env bash

FORMAT=$1
FILE=$2

(
	pushd $(dirname $FILE)
	jupyter nbconvert --to $FORMAT $(basename $FILE)
	popd
)

