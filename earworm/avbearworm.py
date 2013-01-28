#!/usr/bin/env python
# encoding: utf=8

"""
avbearworm.py

Accepts three songs: two original and one A vs B mashup. Graphs the A vs B mashup according to timbre and pitch features.
Matches nodes in mashup up with nodes in two original songs with common timbre and pitch attributes.

Created by Tristan Jehan and Jason Sundram
Modified by Caitlyn Clabaugh
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

def read_graph(name="graph.gpkl"):
    if os.path.splitext(name)[1] == ".gml": 
        return nx.read_gml(name)
    else: 
        return nx.read_gpickle(name)

def save_graph(graph, name="graph.gpkl"):
    if os.path.splitext(name)[1] == ".gml": 
        nx.write_gml(graph, name)
    else: 
        nx.write_gpickle(graph, name)

def print_screen(graph):
    for n,d in graph.nodes_iter(data=True):
        print(n,d)

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

def reconstruct(graph_avb, track_avb, track_a, track_b):
    # TO DO: use song and node label for graph_avb
    # grab section of track_a and track_b

    # grab intro, outro, and total durations
    dur = track_avb.analysis.duration
    dur_intro = min(graph_avb.nodes())
    dur_outro = dur - max(graph_avb.nodes())

    # TO DO: use CAPSULE.PY as template
    for n,d in graph_avb.nodes_iter(data=True):
        if d['song'] == 'a':
            break
        elif d['song'] == 'b':
            break
    return 1

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
    return DG

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

def compute_path(graph, target):

    first_node = min(graph.nodes())
    last_node = max(graph.nodes())
        
    # find the shortest direct path from first node to last node
    if target == 0:
        def dist(node1, node2): return node2-node1 # not sure why, but it works
        # we find actual jumps
        edges = graph.edges(data=True)
        path = tuples(nx.astar_path(graph, first_node, last_node, dist))
        res = collect(edges, path)
        return res
    
    duration = last_node - first_node
    if target < duration: 
        # build a list of sorted jumps by length.
        remaining = duration-target
        # build a list of sorted loops by length.
        loops = get_jumps(graph, mode='forward')
        
        def valid_jump(jump, jumps, duration):
            for j in jumps:
                if j[0] < jump[0] and jump[0] < j[1]:
                    return False
                if j[0] < jump[1] and jump[1] < j[1]:
                    return False
                if duration - (jump[1]-jump[0]+jump[2]['duration']) < 0:
                    return False
            if duration - (jump[1]-jump[0]+jump[2]['duration']) < 0:
                return False
            return True
        
        res = []
        while 0 < remaining:
            if len(loops) == 0: break
            for l in loops:
                if valid_jump(l, res, remaining) == True:
                    res.append(l)
                    remaining -= (l[1]-l[0]+l[2]['duration'])
                    loops.remove(l)
                    break
                if l == loops[-1]:
                    loops.remove(l)
                    break
        res = sorted(res, key=operator.itemgetter(0))
        
    elif duration < target:
        remaining = target-duration
        loops = get_jumps(graph, mode='backward')
        tmp_loops = deepcopy(loops)
        res = []
        # this resolution value is about the smallest denominator
        resolution = loops[-1][1]-loops[-1][0]-loops[-1][2]['duration']
        while remaining > 0:
            if len(tmp_loops) == 0: 
                tmp_loops = deepcopy(loops)
            d = -9999999999999999
            i = 0
            while d < resolution and i+1 <= len(tmp_loops):
                l = tmp_loops[i]
                d = remaining - (l[0]-l[1]+l[2]['duration'])
                i += 1
            res.append(l)
            remaining -= (l[0]-l[1]+l[2]['duration'])
            tmp_loops.remove(l)
        order = np.argsort([l[0] for l in res]).tolist()
        res =  [res[i] for i in order]
        
    else:
        return [(first_node, last_node)]
        
    def dist(node1, node2): return 0 # not sure why, but it works
    start = tuples(nx.astar_path(graph, first_node, res[0][0], dist))
    end = tuples(nx.astar_path(graph, res[-1][1], last_node, dist))
    
    return start + res + end

def label(graph_avb, graph_a, graph_b):
    distances = []
    
    for n,d in graph_avb.nodes_iter(data=True):
        nearest_a, dist_a = findNearest(d, graph_a)
        nearest_b, dist_b = findNearest(d, graph_b)
        if dist_a <= dist_b:
            graph_avb.add_node(n,song = 'a',nearest = nearest_a, dist = dist_a)
            distances.append(dist_a)
        else:
            graph_avb.add_node(n,song = 'b',nearest = nearest_b, dist = dist_b)
            distances.append(dist_b)
   
    return graph_avb, distances

def findNearest(data, graph):
    nearest = None
    curr_dist = 0
    min_dist = float('Inf')
    
    for n,d in graph.nodes_iter(data=True):
        curr_dist = distance(d['timbre'], data['timbre']) + distance(d['pitch'], data['pitch'])
        if curr_dist < min_dist:
            min_dist = curr_dist
            nearest = n

    return nearest, min_dist

def distance(a, b):
    dist = 0

    for i in range(min(len(a),len(b))):
        dist += pow( (a[i]-b[i]), 2)
 
    return sqrt(dist)

def standardDev(dist_list):
    mean = sum(dist_list)/len(dist_list)
    mean_sq_dist = [ pow((x - mean), 2) for x in dist_list ]    
    return sqrt(sum(mean_sq_dist)/len(dist_list))
    
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
    parser.add_option("-r", "--reconstruct", action="store_true", help="reconstruct a vs b track with corresponding parts of track a and b")
    
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
    reconstruct = options.reconstruct
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
    graph_avb, distances = label(graph_avb, graph_a, graph_b)
    if verbose == True:
        print("Summed distance:",sum(distances), "Max distance:",max(distances), "Min distance:",min(distances))
        print("Average:",average(distances), "Standard deviation:",standardDev(distances))
        print_screen(graph_avb)

    if plot == True:
        print("Plotting labelled A vs B graph...")
        sorted_graph = save_plot(graph_avb, mp3_a, mp3_b, mp3_avb+".graph.png")

    if reconstruct == True:
        print("Reconstructing A vs B from A and B...")
        reconstruct(sorted_graph, track_a, track_b)

    print("Completed.")
    return 1


if __name__ == "__main__":
    main()
