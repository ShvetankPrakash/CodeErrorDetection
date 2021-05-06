#!/bin/sh

# Skeleton script to validate the data (code blocks)
# are valid and correct

codeDir="./pythonCode"

for dir in "$codeDir"/*
do
   find $dir -name "*.py" -exec python testChopping.py {} \;
done
