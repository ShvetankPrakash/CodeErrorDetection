#!/bin/bash

# Script to validate the data (code blocks)
# are valid and correct

codeDir="./pythonCode"

validateFileBlocks() { 
   for file in "$@"
   do
      python testChopping.py $file > /dev/null 2>&1
      if [ $? -ne 0 ]; then
         echo FAILED: $file 
      fi
   done
}
export -f validateFileBlocks

for dir in "$codeDir"/*
do
   find $dir -name "*.py" -exec bash -c 'validateFileBlocks "$@"' bash {} +;
   #find $dir -name "*.py" -exec python testChopping.py {} \;
done
