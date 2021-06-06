import sys
import numpy as np
from utils import *

# Dim of square code blocks
SIZE = 80


def chopFile(filename):
   # Open file and create SIZExSIZE char arr to chop code into blocks
   f = open(filename, "r")
   matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
   codeBlocks = []

   char = f.read(1)
   row = 0
   col = 0
   while char != '':
      asciiVal = ord(char)
      matrix[row, col] = asciiVal

      # Increment & read next char
      col += 1
      if char == '\n' or col == SIZE:
         row += 1
         col = 0
      col = col % SIZE
      char = f.read(1)

      # End of code block reached, create new block
      if row == SIZE:
         if isValidPython(codeBlockToString(matrix)):
            codeBlocks.append(matrix) 
            matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
            row = 0
            col = 0
         else:
            nextBlock = np.zeros((0, SIZE), dtype=np.uint8)
            lastRowErased = SIZE
            while not isValidPython(codeBlockToString(matrix)):
               nextBlock = np.insert(nextBlock, 0, matrix[lastRowErased - 1], axis=0)
               matrix[lastRowErased - 1] = 0
               lastRowErased -= 1
               if lastRowErased < 1:
                  #print(len(codeBlocks))
                  #raise Exception("Can not create a valid code block...")
                  return writeBlocks(f, filename, codeBlocks)
                  
            codeBlocks.append(matrix) 
            matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
            row = nextBlock.shape[0]
            col = 0
            matrix[0:row] = nextBlock[0:row]

   if not (matrix == 0).all() and isValidPython(codeBlockToString(matrix)):
      codeBlocks.append(matrix) 

   writeBlocks(f, filename, codeBlocks)


def writeBlocks(f, filename, codeBlocks):      
   # Write chopped blocks to .npy file
   writeName = filename.split("/")[-1].split(".")[0]
   for i, block in enumerate(codeBlocks):
      np.save("./dataset/" + writeName + "_" + str(i) + ".npy", block)  

   f.close()
   return codeBlocks

       
if __name__ == "__main__":
   # filename to be chopped passed as command line arg
   if len(sys.argv) != 2:
      print("Usage: python generateData.py <filename.py>")
   else:  
      chopFile(sys.argv[1])
