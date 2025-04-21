#!/bin/bash

for i in ./imgs/*
do
    if test -f "$i" 
    then
       python3 ./textures.py $i
    fi
done