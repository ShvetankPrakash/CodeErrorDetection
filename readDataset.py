# Script to create dataset for TF from numpy files

import numpy as np
import glob, os, sys

def readDataset(directory):
   # Seed random num generator for reproducability
   np.random.seed(0)

   # Create dataset with labels for training
   dataset = np.zeros((0, 80, 80, 1), dtype=np.uint8)
   labels  = np.zeros((0), dtype = np.uint8)

   os.chdir(directory)
   for filename in glob.glob("*.npy"):
      codeBlock = np.load(filename, allow_pickle=True) 
      codeBlock = np.expand_dims(codeBlock, axis=2)  # add channel-last ordering
      
      # Add error-free code block to dataset
      dataset = np.insert(dataset, dataset.shape[0], codeBlock, 0)
      labels = np.append(labels, 1)

      # Add error to code block and add to dataset
      errorBlock = np.copy(codeBlock)
      indexOne = np.random.randint(79) 
      indexTwo = np.random.randint(79)
      randChar = np.random.randint(255) 

      # Make sure random char is not same as orig char to enforce error
      while randChar == errorBlock[indexOne, indexTwo]: 
         randChar = np.random.randint(255) 
      errorBlock[indexOne, indexTwo] = randChar # set dummy error 
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
