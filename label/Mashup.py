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
import sys
from string import split
from itertools import combinations

from echonest.action import Playback, Jump, Crossfade, Crossmatch, render, display_actions
# from echonest.cloud_support import AnalyzedAudioFile
from Song import Song
from alignment_support import alignment_labeling
from genetic_support import genetic_labeling

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
    def __init__(self, mashup_filename, source_filenames, 
            crossmatch=True, recompute=False, verbose=False):
        
        self.mashup = Song(mashup_filename, recompute, verbose)
        self.sources = [Song(s, recompute, verbose) for s in source_filenames]
        if crossmatch:
            if verbose: print("Crossmatching sources...")
            self.crossmatch_sources(verbose)
        self.labeled = None

    '''
    Crossmatch pairs of sources and add them to sources list
    '''
    def crossmatch_sources(self, verbose=False):
        for pair in combinations(self.sources, 2):
            #get song name info
            s1_path = split(pair[0].mp3_name, "/")
            s2_path = split(pair[1].mp3_name, "/")
            s1_name = s1_path[len(s1_path)-1]
            s2_name = s2_path[len(s2_path)-1]
            s1_s2 = pair[0].mp3_name+"-"+s2_name+"-cross.mp3"
            s2_s1 = pair[1].mp3_name+"-"+s1_name+"-cross.mp3"
            #get song node/length info
            s1_nodes = pair[0].graph.nodes()
            s2_nodes = pair[1].graph.nodes()
            if len(s1_nodes) > len(s2_nodes):
                s1_nodes = s1_nodes[:len(s2_nodes)]
            elif len(s2_nodes) > len(s1_nodes):
                s2_nodes = s2_nodes[:len(s1_nodes)]

            print(s1_nodes, s2_nodes)

            #check is crossmatch song exists
            try:
                f = open(s1_s2)
                f.close()
                if verbose: print("Found crossmatch %s" % s1_s2)
                self.sources.append(Song(s1_s2))
            except:
                try:
                    f = open(s2_s1)
                    f.close()
                    if verbose: print("Found crossmatch %s" % s2_s1)
                    self.sources.append(Song(s2_s1))
                #RENDER
                except:
                    if verbose: print("Found no precomputed crossmatches.")
                    #load tracks
                    if pair[0].track == None: pair[0].load_track()
                    if pair[1].track == None: pair[1].load_track()
                    tracks = (pair[0].track, pair[1].track)
                    lists = (s1_nodes, s2_nodes)
                    render([Crossmatch(tracks,lists)],s1_s2)
                    self.sources.append(Song(s1_s2))


    '''
    Label mashup with sources using...
        "SA" = sequence alignment
        "GA" = genetic algorithm
    '''
    def label(self, algorithm="GA", verbose=False, plot=None,
            size=300, maxgens=100, crossover=0.9, mutation=0.1, optimum=0.0):
        if algorithm == "SA":
            if verbose: print("Labeling %s using sequence alignment..." % self.mashup.mp3_name)
            self.labeled = alignment_labeling(self, verbose)
            return self.labeled
        else:
            if verbose: print("Labeling %s using genetic algorithm..." % self.mashup.mp3_name)
            self.labeled = genetic_labeling(self, verbose, plot, 
                    size, maxgens, crossover, mutation, optimum)
            return self.labeled
    
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
    def reconstruct(self, algorithm, verbose=False):
        # Check that we have loaded track from Echo Nest
        # Create source dictionary
        source_dict = {}
        if self.mashup.track == None: self.mashup.load_track(verbose)
        for s in self.sources:
            if s.track == None:
                s.load_track(verbose)
            source_dict[s.mp3_name] = s.track

        actions = get_actions(self.labeled, source_dict, verbose)
        
        filename = self.mashup.mp3_name+"-"+algorithm+"-reconstructed.mp3"
        render(actions, filename)
       
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
    parser.add_option("-x", "--crossmatch", action="store_true", help="crossmatch pairs of source songs")
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    parser.add_option("-f", "--force", action="store_true", help="force recompute graph")
    parser.add_option("-l", "--label", dest="algorithm", help="label mashup using ALGO: 'SA' for sequence alignment or 'GA' for genetic algorithm", metavar="ALGO")
    parser.add_option("-r", "--render", action="store_true", help="reconstruct mashup using source songs")
    parser.add_option("-p", "--plot", dest="plot", help="plot GA's progress to .dat file PLOT", metavar="PLOT")
    parser.add_option("-s", "--size", dest="size", help="SIZE of GA population", metavar="SIZE")
    parser.add_option("-g", "--maxgens", dest="maxgens", help="max number of GENS for GA to run", metavar="GENS")
    parser.add_option("-c", "--crossover", dest="crossover", help="CROSSOVER rate for GA", metavar="CROSSOVER")
    parser.add_option("-m", "--mutation", dest="mutation", help="MUTATION rate for GA", metavar="MUTATION")
    parser.add_option("-o", "--optimum", dest="optimum", help="OPTIMUM for GA", metavar="OPTIMUM")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        print("Enter mashup and source song(s).\n")
        parser.print_help()
        return -1
    if len(args) < 2:
        print("Enter at least one source song for mashup: %s.\n" % args[0])
        parser.print_help()
        return -1

    # OPTIONS
    crossmatch = options.crossmatch
    recompute = options.force
    verbose = options.verbose
    label = options.algorithm
    render = options.render
    plot = options.plot
    #size
    if options.size: size = int(options.size)
    else: size = 300
    #max generations
    if options.maxgens: maxgens = int(options.maxgens)
    else: maxgens = 100
    #crossover rate
    if options.crossover: crossover = float(options.crossover)
    else: crossover = 0.9
    if options.mutation: mutation = float(options.mutation)
    else: mutation = 0.2
    if options.optimum: optimum = float(options.optimum)
    else: optimum = 0.0

    # CREATE Mashup data structure
    mashup = Mashup(args[0], args[1:], crossmatch, recompute, verbose)
    
    # LABEL Mashup using sequence alignment or GA
    if label:
        mashup.label(label, verbose, plot, size, maxgens, crossover, mutation, optimum)
    
    # RECONSTRUCT mashup using labeled source song segments
    if render:
        if label:
            mashup.reconstruct(label,verbose)
        else:
            print("Enter labeling option to use render option.")
            parser.print_help()
            return -1

    return 1

###############################--HELPER FUNCTIONS--##################################

'''
Creates list of Echo Nest actions to reconstruct mashup using source songs
    labeled_mashup = graph for mashup with source labels
    sources = dictionary of source song graphs
                sources['mp3_name'] = track
'''
def get_actions(labeled_mashup, sources, verbose=False):
    # TODO remove single transitions for better quality
    # ie in 92 93 94 100 96 97 remove 100
    
    actions = []

    curr_song = labeled_mashup.node[0]['label'][0]
    curr_start = labeled_mashup.node[0]['label'][1]
    curr_beat = labeled_mashup.node[0]['label'][1]-1
    for n,d in labeled_mashup.nodes_iter(data=True):
        # same song
        print(n,d)
        if curr_song == d['label'][0]:
            if verbose: print("SAME %s and %s" % (curr_song, d['label'][0]))
            # jump within song
            if curr_beat != d['label'][1]-1:
                if verbose: print("JUMP %s %d -> %d" % (curr_song, curr_beat, d['label'][1]))
                actions.append( Jump(sources[curr_song], curr_start,
                    curr_beat, (curr_beat-curr_start)) )
                curr_start = d['label'][1]
            curr_beat = d['label'][1]
        # transition to diff song
        else:
            if verbose: print("TRANSITION %s:%d --> %s:%d" % (curr_song, curr_beat, d['label'][0], d['label'][1]))
            # TODO leave buffers?
            actions.append( Playback(sources[curr_song], curr_start, curr_beat-curr_start) )
            tracks = [sources[curr_song], sources[d['label'][0]]]
            starts = [curr_beat, d['label'][1]]
            actions.append( Crossfade(tracks, starts, 3) )
            curr_song, curr_beat = d['label']

    return actions

if __name__ == "__main__":
    main()
