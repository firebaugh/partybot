#!/usr/bin/env python
# encoding: utf=8

"""
SongGraph.py

Accepts a mp3 and creates a graph of timbre and pitch features using Echo Nest.
Provides functions to print graph, draw graph, create an "earworm" graph.

Created by Caitlyn Clabaugh
Based on:
  - earworm.py by Tristan Jehan and Jason Sundram
"""

#hard coded, need to change for each developer :/
from pyechonest import config
config.ECHO_NEST_API_KEY = "TTAPZNVYMGG5KQBJI"

from optparse import OptionParser
import numpy as np
from numpy.matlib import repmat, repeat
from numpy import sqrt
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

###############################--SONG GRAPH CLASS--##################################

class SongGraph:

    def __init__(self, mp3_filename, recompute = False, verbose = False):
        self.mp3_filename = mp3_filename
        self.mp3_name = split(lower(self.mp3_filename), '.mp3')[0]
        self.graph_filename = self.mp3_name+".graph"
        self.graph = self.run(recompute, verbose)

    def run(self, recompute = False, verbose = False): 
        if verbose:  print("Probing for pre-existing graph %s..." % self.graph_filename)
        if self.load_graph(self.graph_filename) is False or recompute is True:
            if verbose: print("Computing graph for track %s..." % self.mp3_name)
            self.graph = self.compute_graph(verbose)
            if verbose: print("Writing computed graph out to file %s..." % self.graph_filename)
            self.write_graph(self.graph_filename)
        return self.graph

    def load_graph(self, graph_filename):
        try:
            f = open(graph_filename, "r")
            lines = f.readlines()
            f.close()

            # num node lines, num edge lines
            n,e = lines[0].split(" ")
            n = (int(n)*3+1)
            e = int(e) + n
        
            # nodes: (node_index, {'timbre', 'pitch'})
            nodes = []
            for i in range(1,n,3):
                features = {}
                timbre = lines[i+1].split(" ")
                pitch = lines[i+2].split(" ")
                features['timbre'] = [float(j) for j in timbre]
                features['pitch'] = [float(k) for k in pitch]
                nodes.append( (int(lines[i]),features) )
            # edges: (source_node_index, destination_node_index)
            edges = []
            for j in range(n+1,e,2):
                edge = lines[j].split(" ")
                edges.append( (int(edge[0]), int(edge[1])) )

            # make digraph
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)
            return self.graph
        except:
            # No pre-existing graph
            return False
        
    def compute_graph(self, verbose = False):
        # Load audio info from EchoNest
        mp3 = LocalAudioFile(self.mp3_filename, verbose)

        timbre = resample_features(mp3, rate=RATE, feature='timbre')
        timbre['matrix'] = timbre_whiten(timbre['matrix'])
        pitch = resample_features(mp3, rate=RATE, feature='pitches')
        # why not whiten pitch matrix?
    
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
    
        markers = getattr(mp3.analysis, timbre['rate'])[timbre['index']:timbre['index']+len(paths)]
        self.graph = make_graph(paths, markers, timbre['matrix'], pitch['matrix'])
    
        # remove last node because empty?
        size = self.graph.number_of_nodes()
        self.graph.remove_node(size-1)
        
        return self.graph
 
    def write_graph(self, graph_filename):
        f = open(graph_filename, "w")
        f.write(str(self.graph.number_of_nodes())+" "+str(self.graph.number_of_edges())+"\n")
        # write node data out to file
        for n,d in self.graph.nodes_iter(data=True):
            f.write(str(n)+"\n")
            f.write(" ".join(str(i) for i in d['timbre']))
            f.write("\n")
            f.write(" ".join(str(j) for j in d['pitch']))
            f.write("\n")
        # write edge data out to file
        f.write("\n".join( (str(s)+" "+str(t)) for s,t in self.graph.edges_iter() ))
        f.close()

    def __print__(self):
        for n,d in self.graph.nodes_iter(data=True):
            print(n,d)

###############################--MAIN--##################################

def main():
    # Command line options
    usage = "usage: %s [options] <path_to_mp3>" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    parser.add_option("-r", "--recompute", action="store_true", help="force recompute graph")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        parser.print_help()
        return -1
    recompute = options.recompute
    verbose = options.verbose

    SongGraph(args[0], recompute, verbose)

    return 1

###############################--EARWORM HELPER FUNCTIONS--##################################

## Print nodes and edges to screen, for debugging
def print_screen(graph):
    for n,d in graph.nodes_iter(data=True):
        print(n,d)
    for e in graph.edges_iter():
        print(e)

## Make directed, earworm-type graph of mp3 with features
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



if __name__ == "__main__":
    main()
