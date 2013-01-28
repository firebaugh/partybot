#!/usr/bin/env python
# encoding: utf=8

"""
avbalignment.py

Accepts three songs: two original songs and one A vs B mashup.
Aligns the original songs with the A vs B mashup according to timbre and pitch features.
Matches nodes using iterative Smith-Waterman local sequence alignment.

Created by Caitlyn Clabaugh
Based on:
  - earworm.py by Tristan Jehan and Jason Sundram
  - alignment.py by Aleksandr Levchuk
"""

#hard coded, need to change for each developer :/
from pyechonest import config
config.ECHO_NEST_API_KEY = "TTAPZNVYMGG5KQBJI"

from copy import deepcopy
from optparse import OptionParser
import numpy as np
from numpy.matlib import repmat, repeat
from numpy import sqrt, average
import operator
import os
import sys
from string import lower, split

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
from echonest.audio import LocalAudioFile
# from echonest.cloud_support import AnalyzedAudioFile

from earworm_support import evaluate_distance, timbre_whiten, resample_features
from utils import rows, tuples, flatten


DEF_DUR = 600
MAX_SIZE = 800
MIN_RANGE = 16
MIN_JUMP = 16
MIN_ALIGN = 16
MAX_EDGES = 8
FADE_OUT = 3
RATE = 'beats'

###############################--EARWORM GRAPH OF TRACK--##################################

## Make directed, earworm-type graph of track with features
def make_graph(paths, markers, timbre_features, pitch_features):
    DG = nx.DiGraph()
    # add nodes
    for i in xrange(len(paths)):
        DG.add_node(markers[i].start, timbre = timbre_features[i], pitch = pitch_features[i])
    # add edges
    edges = []
    for i in xrange(len(paths)):
        if i != len(paths)-1:
            edges.append((markers[i].start, markers[i+1].start, {'distance':0, 'duration': markers[i].duration, 'source':i, 'target':i+1})) # source and target for plots only
        edges.extend([(markers[i].start, markers[l[0]+1].start, {'distance':l[1], 'duration': markers[i].duration, 'source':i, 'target':l[0]+1}) for l in paths[i]])
    DG.add_edges_from(edges)

    # sort by timing
    DG = sort_graph(DG)

    return DG

## Sort directed graph by timing
def sort_graph(graph):
    """save plot with index numbers rather than timing"""
    edges = graph.edges(data=True)
    e = [edge[2]['source'] for edge in edges]
    order = np.argsort(e)
    edges = [edges[i] for i in order.tolist()]
    nodes = graph.nodes(data=True)
    DG = nx.DiGraph()
    for edge in edges:
        source = edge[2]['source']
        target = edge[2]['target']
        DG.add_node(source, timbre = nodes[source][1]['timbre'],
                    pitch = nodes[source][1]['pitch'])
        DG.add_edge(source, target)
    return DG

## Save png image of labeled, directed, earworm-type graph of mashup
def save_plot(graph, track_a, track_b, name="graph.png"):
    """save plot with index numbers rather than timing"""
    edges = graph.edges(data=True)
    e = [edge[2]['source'] for edge in edges]
    order = np.argsort(e)
    edges = [edges[i] for i in order.tolist()]
    nodes = graph.nodes(data=True)
    DG = nx.DiGraph()
    for edge in edges:
        source = edge[2]['source']
        target = edge[2]['target']
        v = target-source-1
        # A
        if nodes[source][1]['song'] == 'a':
            DG.add_node(source, color = 'red', song = 'a',
                        nearest = nodes[source][1]['nearest'],
                        dist = nodes[source][1]['dist'])
            DG.add_edge(source, target)
        # B 
        if nodes[source][1]['song'] == 'b':
            DG.add_node(source, color = 'blue', song = 'b',
                        nearest = nodes[source][1]['nearest'],
                        dist = nodes[source][1]['dist'])
            DG.add_edge(source, target)
    A = nx.to_agraph(DG)
    A.layout()
    A.draw(name)
    return DG

## Prints track graph out to screen
def print_screen(graph):
    for n,d in graph.nodes_iter(data=True):
        print(n,d)

def make_similarity_matrix(matrix, size=MIN_ALIGN):
    singles = matrix.tolist()
    points = [flatten(t) for t in tuples(singles, size)]
    numPoints = len(points)
    # euclidean distance
    distMat = sqrt(np.sum((repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2, axis=1, dtype=np.float32))
    return distMat.reshape((numPoints, numPoints))

def get_paths(matrix, size=MIN_RANGE):
    mat = make_similarity_matrix(matrix, size=MIN_ALIGN)
    paths = []
    for i in xrange(rows(mat)):
        paths.append(get_loop_points(mat[i,:], size))
    return paths

def get_paths_slow(matrix, size=MIN_RANGE):
    paths = []
    for i in xrange(rows(matrix)-MIN_ALIGN+1):
        vector = np.zeros((rows(matrix)-MIN_ALIGN+1,), dtype=np.float32)
        for j in xrange(rows(matrix)-MIN_ALIGN+1):
            vector[j] = evaluate_distance(matrix[i:i+MIN_ALIGN,:], matrix[j:j+MIN_ALIGN,:])
        paths.append(get_loop_points(vector, size))
    return paths

# can this be made faster?
def get_loop_points(vector, size=MIN_RANGE, max_edges=MAX_EDGES):
    res = set()
    m = np.mean(vector)
    s = np.std(vector)
    for i in xrange(vector.size-size):
        sub = vector[i:i+size]
        j = np.argmin(sub)
        if sub[j] < m-s and j != 0 and j != size-1 and sub[j] < sub[j-1] and sub[j] < sub[j+1] and sub[j] != 0:
            res.add((i+j, sub[j]))
            i = i+j # we skip a few steps
    # let's remove clusters of minima
    res = sorted(res, key=operator.itemgetter(0))
    out = set()
    i = 0
    while i < len(res):
        tmp = [res[i]]
        j = 1
        while i+j < len(res):
            if res[i+j][0]-res[i+j-1][0] < MIN_RANGE:
                tmp.append(res[i+j])
                j = j+1
            else:
                break
        tmp = sorted(tmp, key=operator.itemgetter(1))
        out.add(tmp[0])
        i = i+j
    out = sorted(out, key=operator.itemgetter(1))
    return out[:max_edges]

def path_intersect(timbre_paths, pitch_paths):
    assert(len(timbre_paths) == len(pitch_paths))
    paths = []
    for i in xrange(len(timbre_paths)):
        t_list = timbre_paths[i]
        p_list = pitch_paths[i]
        t = [l[0] for l in t_list]
        p = [l[0] for l in p_list]
        r = filter(lambda x:x in t,p)
        res = [(v, t_list[t.index(v)][1] + p_list[p.index(v)][1]) for v in r]
        paths.append(res)
    return paths

##############################--SMITH-WATERMAN ALGO--################################

## Label mashup track graph with track A or track B where appropriate
def label(graph_avb, graph_a, graph_b):
    tmp_graph = deepcopy(graph_avb) #used for removing aligned nodes
    label_graph = deepcopy(graph_avb) #used for labeling
        
    # Loop that keeps finding local alignments until completely labeled
    while(tmp_graph.size() != 0):
        # Use Smith-Waterman to find best local alignment in tmp_graph
        # tmp_a/b contains that nodes that are left to label given the alignment
        # align_a/b is the labeled graph so far, given the resulting alignment
        # score_a/b are the scores of the resulting alignments
        align_a, tmp_a, score_a = water(label_graph, tmp_graph, graph_a)
        align_b, tmp_b, score_b = water(label_graph, tmp_graph, graph_b)
        # Use best score between tracks A and B
        if score_a > score_b:
            label_graph, tmp_graph = align_a, tmp_a
        else:
            label_graph, tmp_graph = align_b, tmp_b
   
    return label_graph

match_award = 2
mismatch_penalty = -3
gap_penalty = -7
ext_gap_penalty = -1
## Calculate score between seq1 and seq2 sites a and b
def match_score(a, b):
    if len(a[1].keys()) == 0 or len(b[1].keys()) == 0: #TO DO: check if empty
        #print("gap")
        return ext_gap_penalty
    else:
        #print(a[1]['timbre'][0], b[1]['timbre'][0])
        d = distance(a[1]['timbre'], b[1]['timbre']) + distance(a[1]['pitch'], b[1]['pitch'])
        #print(d)
        if d == 0:
            print(a[1]['timbre'][0],"match")
            return match_award
        else:
            #print("mismatch")
            return mismatch_penalty

# zeros() was origianlly from NumPy.
# This version is implemented by alevchuk 2011-04-10
def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval

## Trace-back aligned sequences
def finalize(align1, align2):
    align1 = align1[::-1] #reverse sequence 1
    align2 = align2[::-1] #reverse sequence 2
    
    i,j = 0,0
    
    #calcuate identity, score and aligned sequeces
    symbol = ''
    found = 0
    score = 0
    identity = 0
    for i in range(0,len(align1)):
        # if two AAs are the same, then output the letter
        if align1[i] == align2[i]:
            symbol = symbol + align1[i]
            identity = identity + 1
            score += match_score(align1[i], align2[i])
    
        # if they are not identical and none of them is gap
        elif align1[i] != align2[i] and align1[i] != '-' and align2[i] != '-':
            score += match_score(align1[i], align2[i])
            symbol += ' '
            found = 0
    
        #if one of them is a gap, output a space
        elif align1[i] == '-' or align2[i] == '-':
            symbol += ' '
            score += gap_penalty
    
    identity = float(identity) / len(align1) * 100
    
    print 'Identity =', "%3.3f" % identity, 'percent'
    print 'Score =', score
    print align1
    print symbol
    print align2

## Smith-Waterman algorithm
def water(final, seq1, seq2):
    seq1_nodes, seq2_nodes = seq1.nodes(data=True), seq2.nodes(data=True)
    m, n = len(seq1_nodes), len(seq2_nodes) # length of two sequences

    # Generate DP table and traceback path pointer matrix
    score = zeros((m+1, n+1)) # the DP table
    pointer = zeros((m+1, n+1)) # to store the traceback path
    
    max_score = 0 # initial maximum score in DP table
    # Calculate DP table and mark pointers
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diagonal = score[i-1][j-1] + match_score(seq1_nodes[i-1], seq2_nodes[j-1])
            score_up = score[i][j-1] + gap_penalty
            score_left = score[i-1][j] + gap_penalty
            score[i][j] = max(0,score_left, score_up, score_diagonal)
            if score[i][j] == 0:
                pointer[i][j] = 0 # 0 means end of the path
            if score[i][j] == score_left:
                pointer[i][j] = 1 # 1 means trace up
            if score[i][j] == score_up:
                pointer[i][j] = 2 # 2 means trace left
            if score[i][j] == score_diagonal:
                pointer[i][j] = 3 # 3 means trace diagonal
            if score[i][j] >= max_score:
                max_i = i
                max_j = j
                max_score = score[i][j];
    
    align1, align2 = '', '' # initial sequences
    
    i,j = max_i,max_j # indices of path starting point
    
    #traceback, follow pointers
    while pointer[i][j] != 0:
        if pointer[i][j] == 3:
            align1 += seq1[i-1] #TO DO: make graphs, nodes should be flagged as gap or not
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif pointer[i][j] == 2:
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1
        elif pointer[i][j] == 1:
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1

    finalize(align1, align2)


#####################################--MAIN--#############################################

## Euclidean distance function
def distance(a, b):
    dist = 0
    for i in range(min(len(a),len(b))):
        dist += pow( (a[i]-b[i]), 2)
    return sqrt(dist)

## Standard deviation function
def standardDev(dist_list):
    mean = sum(dist_list)/len(dist_list)
    mean_sq_dist = [ pow((x - mean), 2) for x in dist_list ]    
    return sqrt(sum(mean_sq_dist)/len(dist_list))
    
## Do all timbre and pitch analysis of single track
def analyze(track, options):
    vbs = bool(options.verbose)    

    print("Computing resampled and normalized matrices...")
    timbre = resample_features(track, rate=RATE, feature='timbre')
    timbre['matrix'] = timbre_whiten(timbre['matrix'])
    pitch = resample_features(track, rate=RATE, feature='pitches')
    # why not whiten pitch matrix?
    
    print("Computing timbre and pitch paths...")
    # pick a tradeoff between speed and memory size
    if rows(timbre['matrix']) < MAX_SIZE:
        # faster but memory hungry. For euclidean distances only.
        t_paths = get_paths(timbre['matrix'])
        p_paths = get_paths(pitch['matrix'])
    else:
        # slower but memory efficient. Any distance possible.
        t_paths = get_paths_slow(timbre['matrix'])
        p_paths = get_paths_slow(pitch['matrix'])
    
    # intersection of top timbre and pitch results
    paths = path_intersect(t_paths, p_paths)
    
    print("Creating graph...")
    markers = getattr(track.analysis, timbre['rate'])[timbre['index']:timbre['index']+len(paths)]
    return make_graph(paths, markers, timbre['matrix'], pitch['matrix'])

def main():
    usage = "usage: %s [options] <a_vs_b_mp3> <a_mp3> <b_mp3>" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    parser.add_option("-p", "--plot", action="store_true", help="plot a colored png graph")
    
    # Print help
    (options, args) = parser.parse_args()
    if len(args) < 3:
        parser.print_help()
        return -1

    mp3_avb = split(lower(args[0]), '.mp3')[0]
    mp3_a = split(lower(args[1]), '.mp3')[0]
    mp3_b = split(lower(args[2]), '.mp3')[0]

    print("Loading audio files...") #why does this take so frickin long?
    verbose = options.verbose
    plot = options.plot
    track_avb = LocalAudioFile(args[0], verbose=verbose)
    track_a = LocalAudioFile(args[1], verbose=verbose)
    track_b = LocalAudioFile(args[2], verbose=verbose)

    print("Analyzing A vs B track...")
    graph_avb = analyze(track_avb, options)
    print("Analyzing A track...")
    graph_a = analyze(track_a, options)
    print("Analyzing B track...")
    graph_b = analyze(track_b, options)
    print("Finished computing all graphs.")

    print("Labeling A vs B based on feature distance from A or B...")
    graph_avb_align = label(graph_avb, graph_a, graph_b)
    '''if verbose == True:
        print("Summed distance:",sum(distances), "Max distance:",max(distances), "Min distance:",min(distances))
        print("Average:",average(distances), "Standard deviation:",standardDev(distances))
        print_screen(graph_avb)'''

    if plot == True:
        print("Plotting labeled A vs B graph...")
        sorted_graph = save_plot(graph_avb, mp3_a, mp3_b, mp3_avb+".graph.png")

    print("Completed.")
    return 1


if __name__ == "__main__":
    main()
