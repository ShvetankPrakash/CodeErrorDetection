import sys
import numpy as np


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
         codeBlocks.append(matrix) 
         matrix = np.zeros((SIZE, SIZE), dtype=np.uint8)
         row = 0
         col = 0

   if not (matrix == 0).all():
      codeBlocks.append(matrix) 
      

   # Write chopped blocks to .npy file
   # TODO
   # print(codeBlocks)

   f.close()
   return codeBlocks

       

if __name__ == "__main__":
   # filename to be chopped passed as command line arg
   if len(sys.argv) != 2:
      print("Usage: python generateData.py <filename.py>")
   else:  
      chopFile(sys.argv[1])
