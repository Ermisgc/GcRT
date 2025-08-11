#!/bin/bash
for i in {1..10}
do
    ./prof.sh ./bin/Debug/sumarray.exe unified
    ./prof.sh ./bin/Debug/sumarray.exe 
    ./prof.sh ./bin/Debug/sumarray.exe pinned
done