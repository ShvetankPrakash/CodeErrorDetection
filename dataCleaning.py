# Clean up "bad" data
import sys

def is_ascii(filename):
   f = open(filename, "r")
   s = f.read() 
   return all(ord(c) < 128 for c in s)


def is_bad_file(filename):
   ascii_only = is_ascii(filename)

   if not ascii_only:
      print("BAD FILE.")

if __name__ == "__main__":
   # filename to be checked as command line arg
   if len(sys.argv) != 2:
      print("Usage: python dataCleaning.py <filename.py>")
   else:  
      is_bad_file(sys.argv[1])
