# Script to create dataset for TF from numpy files

import numpy as np
import glob, os, sys

def readDataset(directory):
   # Create dataset with labels for training
   dataset = np.zeros((0, 80, 80), dtype=np.uint8)
   labels  = np.zeros((0), dtype = np.uint8)

   os.chdir(directory)
   for filename in glob.glob("*.npy"):
      codeBlock = np.load(filename, allow_pickle=True) 
      
      # Add error-free code block to dataset
      dataset = np.insert(dataset, dataset.shape[0], codeBlock, 0)
      labels = np.append(labels, 1)

      # Add error to code block and add to dataset
      errorBlock = codeBlock 
      errorBlock[0,0] = ord('!')  # dummy error for now (TODO: FIX)
      dataset = np.insert(dataset, dataset.shape[0], errorBlock, 0)
      labels = np.append(labels, 0)
  
   print(dataset.shape)
   print(labels.shape) 
   return dataset, labels
      

if __name__ == "__main__":
   # directory containing .npy files for dataset passed as command line arg
   if len(sys.argv) != 2:
      print("Usage: python readDataset.py <dataset_dir>")
   else:  
      readDataset(sys.argv[1])
