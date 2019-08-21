#!/usr/bin/env bash
# copy readme but remove title
tail -n +2 ../README.md > index.md+
nanosite build
