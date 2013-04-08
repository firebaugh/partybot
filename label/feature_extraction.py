#/usr/bin/env python
# encoding: utf=8

"""
feature_extraction.py

Accepts mp3s and extracts and writes out to file a set of statistical music (for both timbre and pitch) features:
- average
- standard deviation
- variance
- skew
- kurtosis

Created by Caitlyn Clabaugh
"""

import numpy as np
import scipy.stats as sp
import sys
from optparse import OptionParser
from Song import Song

def main():
    # Command line options
    usage = "usage: %s [options] [<path_to_song1> <path_to_song2> ...]" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    parser.add_option("-f", "--force", action="store_true", help="force recompute graph")
    parser.add_option("--out", dest="out_file", help="write results to path_to_song_OUT.vec", metavar="OUT")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        print("Enter at least one song for feature extraction.\n")
        parser.print_help()
        return -1

    # OPTIONS
    verbose = options.verbose
    recompute = options.force
    if options.out_file: out = '_'+options.out_file+'.vec'
    else: out = '.vec'

    # EXTRACT FEATURES
    if verbose: print("Analyzing song graphs...")
    songs = [Song(a, recompute, verbose) for a in args]
    for s in songs:
        if verbose: print("Extracting features for %s..." % s.mp3_name)
        timbre, pitch = [], []
        for n,d in s.graph.nodes_iter(data=True):
            #NOTE average of averages?
            timbre.append(np.mean(d['timbre']))
            pitch.append(np.mean(d['pitch']))
        # WRITE OUT TO FILE
        filename = s.mp3_path+out
        if verbose: print("Writing feature vector out to %s..." % filename)
        f = open(filename, "w")
        # timbre_mean, timbre_std, timbre_var, timbre_skew, timbre_kurtosis,
        # pitch_mean, pitch_std, pitch_var, pitch_skew, pitch_kurtosis
        f.write(','.join([str(np.mean(timbre)), str(np.std(timbre)), str(np.var(timbre)), str(sp.skew(timbre)), str(sp.kurtosis(timbre)), str(np.mean(pitch)), str(np.std(pitch)), str(np.var(pitch)), str(sp.skew(pitch)), str(sp.kurtosis(pitch))]))
        f.close()
        if verbose: print("Completed feature extraction for %s" % s.mp3_name)

if __name__ == "__main__":
    main()
