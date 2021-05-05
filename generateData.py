import sys
import numpy as np

def chopFile(filename):
   f = open(filename, "r")
   print(f.read(80*80))

if __name__ == "__main__":
   # filename to be chopped passed as command line arg
   if len(sys.argv) != 2:
      print("Usage: python generateData.py <filename.py>")
   else:  
      chopFile(sys.argv[1])
