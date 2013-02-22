#/usr/bin/env python
# encoding: utf=8

"""
Mashup.py

Accepts mp3s, mashup and source songs, and stores graphs of them using Song.py w/ Echo Nest.
Functionality:
- print graphs to screen
- draw mashup graph as .png w/ or w/o labels
- label mashup with corresponding segments of source congs using either:
    *sequence alignment
    *genetic algorithm (default)
- render mashup track to .mp3
- reconstruct mashup with labeled source song segments

Created by Caitlyn Clabaugh
"""

#hard coded, need to change for each developer :/
from pyechonest import config
config.ECHO_NEST_API_KEY = "TTAPZNVYMGG5KQBJI"

from optparse import OptionParser
import numpy as np
import sys

try:
    import networkx as nx
except ImportError:
    print """earworm.py requires networkx. 
    
If setuptools is installed on your system, simply:
easy_install networkx 

Otherwise, you can get it here: http://pypi.python.org/pypi/networkx

Get the source, unzip it, cd to the directory it is in and run:
    python setup.py install
"""
    sys.exit(1)

from echonest.action import Playback, Jump, Fadeout, render, display_actions
# from echonest.cloud_support import AnalyzedAudioFile
from Song import Song
from alignment_support import align

###############################--MASHUP CLASS--##################################

class Mashup:

    '''
    Associates mashup with its source songs and reconstructs mashup using source segments.
        mashup = graph of input mashup
        sources = graphs of input source songs
        labeled = graph of labeled mashup
        recompute = force recompute graphs for tracks
        verbose = print results on screen
    '''
    def __init__(self, mashup_filename, source_filenames, recompute = False, verbose = False):
        
        self.mashup = Song(mashup_filename, recompute, verbose)
        self.sources = [Song(s, recompute, verbose) for s in source_filenames]
        self.labeled = None

    '''
    Label mashup with sources using...
        "SA" = sequence alignment
        "GA" = genetic algorithm
    '''
    def label(self, algorithm = "GA", verbose = False):
        if algorithm == "SA":
            if verbose: print("Labeling %s using sequence alignment..." % self.mashup.mp3_name)
            self.labeled = align(self, verbose)
            return self.labeled
        else:
            #TODO use GA
            if verbose: print("Labeling %s using genetic algorithm..." % self.mashup.mp3_name)
            return 0
    
    '''
    Evaluate mashup versus labeled mashup
    '''
    def eval(self):
        print("== LABELED GRAPH ==")
        for n,d in self.labeled.nodes_iter(data=True):
            print(n,d)

    '''
    Render reconstruction of mashup using labeled segments of sources
    '''
    def reconstruct(self, mp3_filename):
        # Check that we have loaded track from Echo Nest
        if self.track == None:
            self.load_track(True)

        # NOTE to shorten/lengthen refer to compute_path() in earworm.py
        # renders full length of song
        actions = [Playback(self.track, min(self.graph.nodes()), max(self.graph.nodes()))]
        render(actions, mp3_filename) 
       
    '''
    Print mashup, sources, and labeled mashup to screen
    '''
    def __repr__(self):
        print("GRAPH FOR MASHUP: %s" % self.mashup.mp3_name)
        for n,d in self.mashup.graph.nodes_iter(data=True):
            print(n,d)
        for s in self.sources:
            print(s)
            print("GRAPH FOR SOURCE: %s" % s.mp3_name)
            for n,d in s.graph.nodes_iter(data=True):
                print(n,d)
        return "" #TODO I dont think this is orthodox

###############################--MAIN--##################################

def main():
    # Command line options
    usage = "usage: %s [options] <path_to_mashup> [<path_to_source1> <path_to_source2> ...]" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    parser.add_option("-r", "--recompute", action="store_true", help="force recompute graph")
    parser.add_option("-l", "--label", dest="algorithm", help="label mashup using ALGORITHM", metavar="ALGO")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        print("Enter mashup and source song(s).\n")
        parser.print_help()
        return -1
    if len(args) < 2:
        print("Enter at least one source song for mashup: %s.\n" % args[0])
        parser.print_help()
        return -1

    recompute = options.recompute
    verbose = options.verbose
    labeling = options.algorithm
    
    mashup = Mashup(args[0], args[1:], recompute, verbose)
    
    if labeling: mashup.label(labeling, verbose)

    return 1

if __name__ == "__main__":
    main()
