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

from optparse import OptionParser
import numpy as np
from numpy.matlib import repmat, repeat
from numpy import sqrt, average
import operator
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

## Prints track graph and data out to screen
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
def label(graph_avb, graph_a, graph_b, verbose):
    unlabeled = graph_avb.nodes()

    # Loop until whole graph completely labeled
    while(len(unlabeled) > 0):
        # Use Smith-Waterman to find best local alignment
        # unlabeled_a/b are lists of unalabeled node indexes
        # label_a/b is segment of graph_avb or lobal alignment
        # score_a/b are the scores of alignments with song a/b
        label_a, unlabeled_a, score_a = water(graph_avb, unlabeled, graph_a, 'a')
        label_b, unlabeled_b, score_b = water(graph_avb, unlabeled, graph_b, 'b')
        # Use best score between tracks A and B
        if score_a > score_b:
            graph_avb = label_a.to_directed()
            unlabeled = unlabeled_a
            if verbose: print("Found song a segment. Remaining:", len(unlabeled))
        else:
            graph_avb = label_b.to_directed()
            unlabeled = unlabeled_b
            if verbose: print("Found song b segment. Remaining:", len(unlabeled))
   
    return graph_avb

match_award = 5 
gap_penalty = -5
#cut_cost = -7
## Calculate score between seq1 and seq2 sites a and b
def match_score(a, b):
    if not a or not b: #TODO how to account for gaps?
        return gap_penalty
    else:
        d = (distance(a['timbre'], b['timbre']) + distance(a['pitch'], b['pitch'])) / 2.0
        if d <= 0.001:
            return match_award
        else:
            return match_award-d #TODO discount dependent on max distance

# zeros() was origianlly from NumPy.
# This version is implemented by alevchuk 2011-04-10
def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval

## Smith-Waterman algorithm
## Modified to take care of graph-specific labeling
def water(final, seq1, seq2, label):
    m, n = len(seq1), seq2.number_of_nodes() # length of two sequences

    # Generate DP table and traceback path pointer matrix
    score = zeros((m+1, n+1)) # the DP table
    pointer = zeros((m+1, n+1)) # to store the traceback path
    
    max_score = 0 # initial maximum score in DP table
    # Calculate DP table and mark pointers
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diagonal = score[i-1][j-1] + match_score(final.node[seq1[i-1]], seq2.node[j-1])
            score_up = score[i][j-1] + gap_penalty
            score_left = score[i-1][j] + gap_penalty
            score[i][j] = max(0, score_left, score_up, score_diagonal)
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
    
    align1, align2 = [], []

    i,j = max_i,max_j # indices of path starting point
    
    #traceback, follow pointers
    while pointer[i][j] != 0:
        if pointer[i][j] == 3:
            align1.append(seq1[i-1])
            align2.append(j-1)
            i -= 1
            j -= 1
        elif pointer[i][j] == 2:
            align1.append(seq1[i])
            align2.append(j-1)
            j -= 1
        elif pointer[i][j] == 1:
            align1.append(seq1[i-1])
            align2.append(j)
            i -= 1

    #---Labeling specific code---#

    #calculate score and label graph
    align1 = align1[::-1] #reverse sequence 1
    align2 = align2[::-1] #reverse sequence 2

    score = 0
    for i in range(0,len(align1)):
        if align1[i] == None or align2[i] == None:
            score += gap_penalty
        else:
            score += match_score(final.node[align1[i]], seq2.node[align2[i]])
        final.node[align1[i]]['label'] = (label, align2[i])

    #remove labeled nodes 
    seq1 = list(set(seq1)-set(align1))

    return final, seq1, score

#####################################--MAIN--#############################################

## Feature distance between original and labeled mashups
## Returns list of distances between each node
def final_feature_dist(avb, a, b, l, feature):
    distances = []
    avb_nodes, label_nodes = avb.nodes(data=True), l.nodes(data=True)
    n = min([len(avb_nodes),len(label_nodes)])

    for i in range(n-1):
        label = label_nodes[i][1]['label'][0]
        index = label_nodes[i][1]['label'][1]
        
        if label == 'a':
            distances.append(distance(avb_nodes[i][1][feature], a.node[index][feature]))
        elif label == 'b':
            distances.append(distance(avb_nodes[i][1][feature], b.node[index][feature]))

    return distances

## Euclidean distance function between two lists
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
def analyze(trackname, options):
    vbs = bool(options.verbose)    
    
    # Load audio info from EchoNest
    print("Loading %s audio files..." % trackname)
    track = LocalAudioFile(trackname, verbose=vbs)

    if vbs: print("Computing resampled and normalized matrices...")
    timbre = resample_features(track, rate=RATE, feature='timbre')
    timbre['matrix'] = timbre_whiten(timbre['matrix'])
    pitch = resample_features(track, rate=RATE, feature='pitches')
    # why not whiten pitch matrix?
    
    if vbs: print("Computing timbre and pitch paths...")
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
    
    if vbs: print("Creating graph...")
    markers = getattr(track.analysis, timbre['rate'])[timbre['index']:timbre['index']+len(paths)]
    graph = make_graph(paths, markers, timbre['matrix'], pitch['matrix'])
    
    # remove last node because empty?
    size = graph.number_of_nodes()
    graph.remove_node(size-1)
        
    return graph

def read_graph(filename):
    try:
        # Check for pre-existing avb graph
        print("Probing for pre-exisiting graph...")
        f = open(filename, "r")
        f.close()
        print("Using pre-exisiting graph %s." % filename)
    except:
        # If no pre-existing graph
        return False
        
def write_graph(graph, outfile, options):
    print("Writing A vs B graph out to %s..." % outfile)
    f = open(outfile, "w")
     
    f.close()

def main():
    # Command line options
    usage = "usage: %s [options] <a_vs_b_mp3> <a_mp3> <b_mp3>" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    parser.add_option("-p", "--plot", action="store_true", help="plot a colored png graph")
    parser.add_option("-r", "--recompute", action="store_true", help="force recompute graph")
    (options, args) = parser.parse_args()
    if len(args) < 3:
        parser.print_help()
        return -1
    plot = options.plot
    verbose = options.verbose
    recompute = options.recompute

    # Song names
    mp3_avb = split(lower(args[0]), '.mp3')[0]
    mp3_a = split(lower(args[1]), '.mp3')[0]
    mp3_b = split(lower(args[2]), '.mp3')[0]

    '''
    FEATURE EXTRACTION
    ------------------
    Use Echo Nest or pre-existing *-graph.txt to graph musical events
    Each node has two 12 vectors for timbre and pitch features
    '''
    # A VS B GRAPH
    mp3_avb_graph = mp3_avb+".graph"
    # Check for pre-existing avb graph
    print("Probing for pre-exisiting graph for track %s..." % mp3_avb)    
    graph_avb = read_graph(mp3_avb_graph)
    if graph_avb == False:
        print("No existing graph file found.")
        graph_avb = analyze(args[0], options)
        write_graph(graph_avb, mp3_avb_graph, options)
    elif recompute == True:
        print("Force recomputing graph...")
        graph_avb = analyze(args[0], options)
    else:
        print("Using pre-exisiting graph %s." % mp3_avb_graph)
        graph_avb = read_graph(mp3_avb_graph)
    # A GRAPH
    mp3_a_graph = mp3_a+".graph"
    # Check for pre-existing a graph
    print("Probing for pre-exisiting graph for track %s..." % mp3_a)    
    graph_a = read_graph(mp3_a_graph)
    if graph_a == False:
        print("No existing graph file found.")
        graph_a = analyze(args[1], options)
        write_graph(graph_a, mp3_a_graph, options)
    elif recompute == True:
        print("Force recomputing graph...")
        graph_a = analyze(args[1], options)
    else:
        print("Using pre-exisiting graph %s." % mp3_a_graph)
        graph_a = read_graph(mp3_a_graph)
    # B GRAPH
    mp3_b_graph = mp3_b+".graph"
    # Check for pre-existing b graph
    print("Probing for pre-exisiting graph for track %s..." % mp3_b)    
    graph_b = read_graph(mp3_b_graph)
    if graph_b == False:
        print("No existing graph file found.")
        graph_b = analyze(args[2], options)
        write_graph(graph_b, mp3_b_graph, options)
    elif recompute == True:
        print("Force recomputing graph...")
        graph_b = analyze(args[2], options)
    else:
        print("Using pre-exisiting graph %s." % mp3_b_graph)
        graph_b = read_graph(mp3_b_graph)
    print("Finished computing all graphs.")
   

    '''
    LABELLING MASHUP
    Use sequence alignment to create a labeled mashup graph
    Each mashup event will be labeled with its corresponding event
    in either track A or track B
    '''
    print("Labeling A vs B based on feature distance from A or B...")
    graph_label = label(graph_avb, graph_a, graph_b, verbose)
    
    if verbose == True:
        print("== LABELED GRAPH ==")
        print_screen(graph_label)

        pitch_distances = final_feature_dist(graph_avb, graph_a, graph_b, graph_label, 'pitch')
        timbre_distances = final_feature_dist(graph_avb, graph_a, graph_b, graph_label, 'timbre')
        print("== PITCH RESULTS ==")
        print("\tSummed Distance:",sum(pitch_distances))
        print("\tMax Distance:",max(pitch_distances))
        print("\tMin Distance:",min(pitch_distances))
        print("\tAverage Distance:",average(pitch_distances))
        print("\tStandard Dev:",standardDev(pitch_distances))
        
        print("== TIMBRE RESULTS ==")
        print("\tSummed Distance:",sum(timbre_distances))
        print("\tMax Distance:",max(timbre_distances))
        print("\tMin Distance:",min(timbre_distances))
        print("\tAverage Distance:",average(timbre_distances))
        print("\tStandard Dev:",standardDev(timbre_distances))

    if plot == True:
        print("Plotting labeled A vs B graph...")
        sorted_graph = save_plot(graph_avb, mp3_a, mp3_b, mp3_avb+".graph.png")
    
    print("Completed.")
    return 1


if __name__ == "__main__":
    main()
