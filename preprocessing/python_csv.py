#/Library/Frameworks/Python.framework/Versions/2.7/bin/python

import sys
import csv

with open(sys.argv[1]) as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    print(row[' tweet'])