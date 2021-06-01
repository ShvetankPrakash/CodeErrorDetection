# Helper functions for debugging 

import os
import glob
import sys
import numpy as np

def printCode(codeBlock):
   codeBlock = codeBlock.flatten()
   for letter in codeBlock:
      print(chr(letter), end="")


def diffCodeBlock(b1, b2):
   diff = b1 - b2
   for i, row in enumerate(diff):
      if (row == 0).all():
         continue
      print(i)
      print(row)


