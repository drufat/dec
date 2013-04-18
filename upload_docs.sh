#!/usr/bin/env bash

repo=`mktemp -d`
git clone drufat-github:drufat/dec.git ${repo}

(
    cd ${repo}
    git checkout gh-pages
    git rm -rf .
)

#echo "My GitHub Page" > ${repo}/index.html
sphinx-build -b html doc ${repo}

(
    cd ${repo}
    touch .nojekyll
    git add .
    git commit -a -m "Update documents."
    git push origin gh-pages
)

rm -rf ${tmp}
