#!/usr/bin/env bash
(
    cd notebooks
    for f in ./*.ipynb; do 
        PYTHONPATH=.. runipy $f 
    done
)
