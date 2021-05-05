import sys
import numpy as np
from generateData import chopFile

def testChopFile(filename):
   codeBlocks = chopFile(filename)

   if len(codeBlocks) > 0: 
      print("Code was chopped into " + str(len(codeBlocks)) + " blocks.")
      f = open(filename, 'r')
      for currBlock in codeBlocks: 
         currBlock  = currBlock.flatten()
         for i in currBlock:
            if i == 0:
               continue
            assert(f.read(1) == chr(i))
      
   print("PASSED: All blocks match orig file!")


if __name__ == "__main__":
   # filename to be chopped passed as command line arg
   if len(sys.argv) != 2:
      print("Usage: python testChopping.py <filename.py>")
   else:  
      testChopFile(sys.argv[1])
