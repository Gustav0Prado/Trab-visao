#!/bin/bash

for i in ./imgs/photo_{1..16}.jpg
do
    if test -f "$i" 
    then
       python3 ./textures.py $i
    fi
done