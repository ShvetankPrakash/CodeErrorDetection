#!/bin/bash

# Script to validate the data (code blocks)
# are valid and correct

codeDir="./pythonCode"

validateFileBlocks() { 
   for file in "$@"
   do
      # Make sure file is "good" data, if not remove
      python dataCleaning.py $file > /dev/null 2>&1
      if [ $? -ne 0 ]; then
         echo REMOVING: $file 
         rm $file
         continue
      fi
      
      # Make sure chopped file is correct as orignal
      # testChopping.py chops file and writes it as well
      python testChopping.py $file > /dev/null 2>&1
      if [ $? -ne 0 ]; then
         echo FAILED: $file 
      fi
   done
}
export -f validateFileBlocks

# Iterate through all files in code directory and validate data
for dir in "$codeDir"/*
do
   find $dir -name "*.py" -exec bash -c 'validateFileBlocks "$@"' bash {} +;
done
