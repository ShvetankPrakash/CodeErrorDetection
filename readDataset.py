# Script to create dataset for TF from numpy files

import numpy as np
import glob, os, sys
from utils import *

def readDataset(directory, rgb=False):
   # Seed random num generator for reproducability
   np.random.seed(0)

   # Keep track of number of char changes inserted to find avg
   charChanges = 0.0

   # Create dataset with labels for training
   if rgb:
      dataset = np.zeros((0, 80, 80, 3), dtype=np.uint8)
   else:
      dataset = np.zeros((0, 80, 80, 1), dtype=np.uint8)
   labels  = np.zeros((0), dtype = np.uint8)

   os.chdir(directory)
   for filename in glob.glob("*.npy"):
      codeBlock = np.load(filename, allow_pickle=True) 
      existingIndices = np.nonzero(codeBlock)
      codeBlock = np.expand_dims(codeBlock, axis=2)  # add channel-last ordering
      if rgb:
         codeBlock = np.repeat(codeBlock, 3, axis=2)

      # Add error-free code block to dataset
      dataset = np.insert(dataset, dataset.shape[0], codeBlock, 0)
      labels = np.append(labels, 1)

      # Add error to code block and add to dataset
      errorBlock = np.copy(codeBlock)
      numChars = existingIndices[0].shape[0]
      existingCharsOnly = False
      if not existingCharsOnly:
         indexOne = np.random.randint(79) 
         indexTwo = np.random.randint(79)
      else:
         # Replace an existing char in the code not add to a blank cell
         existingChar = np.random.randint(numChars - 1)
         indexOne = existingIndices[0][existingChar]
         indexTwo = existingIndices[1][existingChar]
     
      # Set dummy error 
      #origChar = errorBlock[indexOne, indexTwo]
      randChar = np.random.randint(32, 127) 
      errorBlock[indexOne, indexTwo] = randChar
      charChanges += 1.0

      # Make sure random char makes the code block contain an error 
      while isValidPython(codeBlockToString(errorBlock)): 
         #errorBlock[indexOne, indexTwo] = origChar # reset to orig  
         if not existingCharsOnly:
            indexOne = np.random.randint(79) 
            indexTwo = np.random.randint(79)
         else:
            existingChar = np.random.randint(numChars - 1)
            indexOne = existingIndices[0][existingChar]
            indexTwo = existingIndices[1][existingChar]
         
         #origChar = errorBlock[indexOne, indexTwo]
         randChar = np.random.randint(32, 127) 
         errorBlock[indexOne, indexTwo] = randChar # set dummy error 
         charChanges += 1.0
      dataset = np.insert(dataset, dataset.shape[0], errorBlock, 0)
      labels = np.append(labels, 0)
  
   print(dataset.shape)
   print(labels.shape) 
   print(charChanges)
   print("Average number of chars changed: " + str(charChanges / (dataset.shape[0]/ 2)))
   return dataset, labels
      

if __name__ == "__main__":
   # directory containing .npy files for dataset passed as command line arg
   if len(sys.argv) != 2:
      print("Usage: python readDataset.py <dataset_dir>")
   else:  
      readDataset(sys.argv[1])
