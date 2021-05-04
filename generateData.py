#!/bin/python3.8

import sys

def chopFile(filename):
   f = open(filename, "r")
   print(f.read())

if __name__ == "__main__":
   # filename to be chopped passed as command line arg
   chopFile(sys.argv[1])
