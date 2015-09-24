#!/usr/bin/env bash

FORMAT=$1
FILE=$2

(
	pushd $(dirname $FILE)
	ipython nbconvert --to $FORMAT $(basename $FILE)
	popd
)

