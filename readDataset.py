import numpy as np
import glob, os

def readDataset(directory):
   os.chdir(directory)
   for filename in glob.glob("*.npy"):
       print(filename)
   

 
if __name__ == "__main__":
   # filename to be chopped passed as command line arg
   if len(sys.argv) != 2:
      print("Usage: python readDataset.py <dataset_dir>")
   else:  
      chopFile(sys.argv[1])
