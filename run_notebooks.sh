#!/usr/bin/env bash
(
    pushd examples
    for f in *.ipynb; do 
        PYTHONPATH=.. runipy -o $f 
    done
	popd
)
