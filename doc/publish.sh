#!/usr/bin/env bash
make html
rsync -r build/html/ $ZISSOU_USER@zissou.infosci.cornell.edu:/var/www/html/convokit/documentation

# publish to my personal site
#rsync -a build/html/ ~/dev/qema.github.io/convokit/
#pushd .
#cd ~/dev/qema.github.io
#nanosite p
#popd
