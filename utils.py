# Helper functions for debugging 

import os
import glob
import sys
import numpy as np
import tensorflow as tf

def printCode(codeBlock):
   codeBlock = codeBlock.flatten()
   for letter in codeBlock:
      print(chr(letter), end="")


