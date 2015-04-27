#!/usr/bin/env bash

function upload_github {

    #repo=`mktemp -d /tmp/doc.XXXXXXXXX`
    repo=./build/html

    if cd ${repo}; then 
        git pull
    else 
        git clone --depth 1 git@github:drufat/dec.git ${repo}
    fi 
    (
        cd ${repo}
        git checkout gh-pages
    )
    
    make
    sphinx-build -b html doc ${repo}
    (
        cd ${repo}
        touch .nojekyll
        git commit -a -m "Update documents."
        git push origin gh-pages
    )
}

function upload_private {

    sphinx-build -b html doc build/html
    rsync -avz --delete build/html/ drufat@pi.dzhelil.info:restric/dec
    
}

upload_private
