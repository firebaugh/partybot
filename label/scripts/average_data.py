#/usr/bin/env python
# encoding: utf=8

"""
average_data.py

Writes out .dat file of averages from .dat file inputs

Created by Caitlyn Clabaugh
"""

import numpy as np
import scipy.stats as sp
import sys
from optparse import OptionParser

def main():
    # Command line options
    usage = "usage: %s [options] <path_to_outfile> <path_to_dat1> <path_to_dat2> [<path_to_dat3> ...]" % sys.argv[0]
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) < 3:
        print("Enter out filename followed by at least two .dat filenames.\n")
        parser.print_help()
        return -1

    # OPTIONS
    out = args[0]

    # READ IN
    files = []
    for a in range(1,len(args)):
        f = open(args[a], "r")
        lines = []
        for line in f.readlines():
            if line[0] != '#': #skip commented lines
                   lines.append(line.split("\t"))
        files.append(lines)
        f.close()

    # ANALYZE
    data = []
    for i in range(len(files)):
        lines = files[i]
        for j in range(len(lines)):
            try:
                data[j][0] += 1 #count
                for k in range(len(lines[j])):
                    data[j][k+1] += float(lines[j][k])
            except:
                els = [1]
                for k in range(len(lines[j])):
                    els.append(float(lines[j][k]))
                data.append(els)

    # WRITE OUT
    f = open(out, "w")
    for d in data:
        f.write("\t".join([str(d[i]/float(d[0])) for i in range(1,len(d))]))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    main()
