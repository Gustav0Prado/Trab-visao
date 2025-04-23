#!/bin/bash

for i in ./imgs/{1..16}.*
do
    if test -f "$i" 
    then
       python3 ./textures.py $i
    fi
done