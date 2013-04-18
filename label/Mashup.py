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
from itertools import combinations

from echonest.remix.action import Playback, Jump, Crossfade, Crossmatch, render
# from echonest.cloud_support import AnalyzedAudioFile
from Song import Song
from alignment_support import alignment_labeling
from genetic_support import genetic_labeling
from earworm_support import equalize_tracks
from report import email

###############################--MASHUP CLASS--##################################

class Mashup:

    '''
    Associates mashup with its source songs and reconstructs mashup using source segments.
        mashup = graph of input mashup
        sources = graphs of input source songs
        labeled = graph of labeled mashup
        crossmatch = crossmatch all combinations of source songs
        recompute = force recompute graphs for tracks
        verbose = print results on screen
    '''
    def __init__(self, mashup_filename, source_filenames, 
            crossmatch=True, recompute=False, verbose=False):
        
        self.mashup = Song(mashup_filename, recompute, verbose)
        self.sources = [Song(s, recompute, verbose) for s in source_filenames]
        if crossmatch:
            if verbose: print("Crossmatching sources...")
            self.crossmatch_sources(recompute, verbose)
        self.labeled = None

    '''
    Crossmatch pairs of sources and add them to sources list
    '''
    def crossmatch_sources(self, recompute=False, verbose=False):
        #[(start,duration),...] by timing rather than node index
        def to_tuples(graph):
            return [(d['source'],d['duration']) for s,t,d in graph.edges_iter(data=True)]

        #for all combinations of source songs
        for pair in combinations(self.sources, 2):
            #get crossmatch filename and beats lists
            s1_s2 = "-".join([pair[0].mp3_path,pair[1].mp3_name,"cross.mp3"])
            s2_s1 = "-".join([pair[1].mp3_path,pair[0].mp3_name,"cross.mp3"])
            s1_beats, s2_beats = to_tuples(pair[0].graph), to_tuples(pair[1].graph)
            #use length of min source
            if len(s1_beats) > len(s2_beats):
                s1_beats = s1_beats[:len(s2_beats)]
            elif len(s2_beats) > len(s1_beats):
                s2_beats = s2_beats[:len(s1_beats)]

            #check if crossmatch mp3 exists
            try:
                f = open(s1_s2)
                f.close()
                if verbose: print("Found precomputed crossmatch %s" % s1_s2)
                if recompute: raise Exception()
                self.sources.append(Song(s1_s2))
            except:
                try:
                    f = open(s2_s1)
                    f.close()
                    if verbose: print("Found precomputed crossmatch %s" % s2_s1)
                    if recompute: raise Exception()
                    self.sources.append(Song(s2_s1))
                #RENDER new crossmatch mp3
                except:
                    if verbose and not recompute: print("Found no precomputed crossmatches.")
                    if verbose and recompute: print("Recomputing crossmatches...")
                    #load tracks
                    if pair[0].track == None: pair[0].load_track()
                    if pair[1].track == None: pair[1].load_track()
                    #equalize tracks
                    #TODO beat match to mashup tempo
                    pair[0].track, pair[1].track = equalize_tracks([pair[0].track,pair[1].track])
                    if verbose: print("Rendering crossmatch %s..." % s1_s2)
                    render([Crossmatch( (pair[0].track,pair[1].track), (s1_beats,s2_beats) )], s1_s2) 
                    self.sources.append(Song(s1_s2))


    '''
    Label mashup with sources using...
        "SA" = sequence alignment
        "GA" = genetic algorithm
    '''
    def label(self, algorithm="GA", verbose=False, out=None,
            size=300, maxgens=100, crossover=0.9, mutation=0.1, optimum=0.0, 
            restrict=False, converge=True, smooth=False):
        if algorithm == "SA":
            if verbose: print("Labeling %s using sequence alignment..." % self.mashup.mp3_name)
            self.labeled = alignment_labeling(self, verbose)
            return self.labeled
        else:
            if verbose: print("Labeling %s using genetic algorithm..." % self.mashup.mp3_name)
            self.labeled = genetic_labeling(self, verbose, out, 
                    size, maxgens, crossover, mutation, optimum,
                    restrict, converge, smooth)
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
    def reconstruct(self, out, algorithm, verbose=False):
        # Check that we have loaded track from Echo Nest
        # Create source dictionary
        source_dict = {}
        if self.mashup.track == None: self.mashup.load_track(verbose)
        for s in self.sources:
            if s.track == None:
                s.load_track(verbose)
            source_dict[s.mp3_name] = s.track

        if verbose: print("Calculatiing actions in reconstructed mashup...")
        actions = get_actions(self.labeled, source_dict, verbose)
       	if verbose: print("Found actions: %s" % actions)
 
        filename = out+"-"+algorithm+"-reconstructed.mp3"
        if verbose: print("Rendering reconstructed mashup...")
        render(actions, filename)
    
    '''
    Write reconstruction out to 2 files: OUT.graph and OUT.segs
    .graph contains normal graph of mashup
    .segs constains: segment# .graph_start_node .graph_stop_node
    '''
    def write_graph(self, out, verbose=False):
        graph_filename = out+".graph"
        segs_filename = out+".segs"

        if verbose: print("Writing out .graph and .segs files for reconstructed mashup...")
        gf = open(graph_filename, "w")
        sf = open(segs_filename, "w")
        gf.write(str(self.labeled.number_of_nodes())+" "+str(self.labeled.number_of_edges())+"\n")
	# write node data out to file
        curr_seg = 0
        start = 0
        prev = self.labeled.node[0]['label'][1]-1
	prev_song = self.labeled.node[0]['label'][0]
        for n,d in self.labeled.nodes_iter(data=True):
            gf.write(str(n)+"\n")
            gf.write(" ".join(str(i) for i in d['timbre']))
            gf.write("\n")
            gf.write(" ".join(str(j) for j in d['pitch']))
            gf.write("\n")
            if d['label'][1] != prev+1:
                sf.write(" ".join([str(curr_seg), prev_song, str(start), str(n-1)]))
                sf.write("\n")
                start = n
                curr_seg += 1
                prev_song = d['label'][0]
            prev = d['label'][1]

        # write edge data out to file
        for s,t,d in self.labeled.edges_iter(data=True):
            gf.write(" ".join([str(s),str(t),
                str(d['duration']),str(d['source']),str(d['target'])]))
            gf.write("\n")
        gf.close()
        sf.close()

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
    parser.add_option("-s", "--smooth", action="store_true", help="converge GA based on exponentially weighted (smoothed) average instead of average.")
    parser.add_option("-c", "--converge", action="store_true", help="run GA until convergence rather than max generations GEN")
    parser.add_option("--restrict", dest="restriction", default=False, help="GA is restricted to aligning with N transitions", metavar="N")
    parser.add_option("--label", dest="algorithm", help="label mashup using ALGO: 'SA' for sequence alignment or 'GA' for genetic algorithm", metavar="ALGO")
    parser.add_option("--size", dest="size", help="SIZE of GA population", metavar="SIZE")
    parser.add_option("--maxgens", dest="maxgens", help="max number of GENS for GA to run", metavar="GENS")
    parser.add_option("--crossover", dest="crossover", help="CROSSOVER rate for GA", metavar="CROSSOVER")
    parser.add_option("--mutation", dest="mutation", help="MUTATION rate for GA", metavar="MUTATION")
    parser.add_option("--optimum", dest="optimum", help="OPTIMUM for GA", metavar="OPTIMUM")
    parser.add_option("--out", dest="out_file", help="write plot of GA's progress to OUT.dat, reconstruction of mashup to OUT-ALGO-reconstructed.mp3, graph of reconstruction to OUT.graph, and corresponding segments of reconstruction to OUT.segs", metavar="OUT")
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
    out = options.out_file
    converge = options.converge
    smooth = options.smooth
    #transition restriction
    if options.restriction: restrict = int(options.restriction)
    else: restrict = 0
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
    if verbose: print("Created mashup data structure.")

    # LABEL Mashup using sequence alignment or GA
    if label:
        mashup.label(label, verbose, out, size, maxgens, crossover, mutation, optimum, restrict, converge, smooth)
        if verbose: print("Completed labeling.")
    
    # PRINT labeled graph
    '''if verbose:
        print("LABELED GRAPH FOR MASHUP: %s" % mashup.mashup.mp3_name)
        for n,d in mashup.labeled.nodes_iter(data=True):
            print(n,d)'''

    # REPORT results
    if label == "GA" and out:
        if verbose: print("Sending out email notification...")
        email(out+'.dat')
    
    # RECONSTRUCT mashup using labeled source song segments
    if render:
        if label and out:
            mashup.reconstruct(out,label,verbose)
            mashup.write_graph(out,verbose)
            if verbose: print("Completed reconstruction.")
        else:
            print("Enter labeling option and out option to render and write out reconstructed mashup.")
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
        if curr_song == d['label'][0]:
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
