#!/usr/bin/env bash

function upload_github {

    repo=`mktemp -d /tmp/doc.XXXXXXXXX`

    git clone git@github:drufat/dec.git ${repo}
    (
        cd ${repo}
        git checkout gh-pages
        git rm -rf .
    )

    sphinx-build -b html doc ${repo}
    (
        cd ${repo}
        touch .nojekyll
        git add .
        git commit -a -m "Update documents."
        git push origin gh-pages
    )

    rm -rf ${tmp}

}

function upload_private {

    sphinx-build -b html doc build/html
    rsync -avz --delete build/html/ drufat@pi.dzhelil.info:restric/dec
    
}

upload_private
