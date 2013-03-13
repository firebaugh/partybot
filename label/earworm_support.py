#!/usr/bin/env python
# encoding: utf-8

"""
earworm_support.py

Created by Tristan Jehan and Jason Sundram.

Caitlyn Clabaugh added functionality from earworm and capsule
"""

import numpy as np
from numpy.matlib import repmat, repeat
from copy import deepcopy
import operator
from utils import rows, tuples, flatten
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

FUSION_INTERVAL = .06   # This is what we use in the analyzer
AVG_PEAK_OFFSET = 0.025 # Estimated time between onset and peak of segment.


def evaluate_distance(mat1, mat2):
    return np.linalg.norm(mat1.flatten() - mat2.flatten())

def timbre_whiten(mat):
    if rows(mat) < 2: return mat
    m = np.zeros((rows(mat), 12), dtype=np.float32)
    m[:,0] = mat[:,0] - np.mean(mat[:,0],0)
    m[:,0] = m[:,0] / np.std(m[:,0],0)
    m[:,1:] = mat[:,1:] - np.mean(mat[:,1:].flatten(),0)
    m[:,1:] = m[:,1:] / np.std(m[:,1:].flatten(),0) # use this!
    return m


def get_central(analysis, member='segments'):
    """ Returns a tuple: 
        1) copy of the members (e.g. segments) between end_of_fade_in and start_of_fade_out.
        2) the index of the first retained member.
    """
    def central(s):
        return analysis.end_of_fade_in <= s.start and (s.start + s.duration) < analysis.start_of_fade_out
    
    members = getattr(analysis, member)
    ret = filter(central, members[:]) 
    index = members.index(ret[0]) if ret else 0
    
    return ret, index


def get_mean_offset(segments, markers):
    if segments == markers:
        return 0
    
    index = 0
    offsets = []
    try:
        for marker in markers:
            while segments[index].start < marker.start + FUSION_INTERVAL:
                offset = abs(marker.start - segments[index].start)
                if offset < FUSION_INTERVAL:
                    offsets.append(offset)
                index += 1
    except IndexError, e:
        pass
    
    return np.average(offsets) if offsets else AVG_PEAK_OFFSET


def resample_features(data, rate='tatums', feature='timbre'):
    """
    Resample segment features to a given rate within fade boundaries.
    @param data: analysis object.
    @param rate: one of the following: segments, tatums, beats, bars.
    @param feature: either timbre or pitch.
    @return A dictionary including a numpy matrix of size len(rate) x 12, a rate, and an index
    """
    ret = {'rate': rate, 'index': 0, 'cursor': 0, 'matrix': np.zeros((1, 12), dtype=np.float32)}
    segments, ind = get_central(data.analysis, 'segments')
    markers, ret['index'] = get_central(data.analysis, rate)
    
    if len(segments) < 2 or len(markers) < 2:
        return ret
        
    # Find the optimal attack offset
    meanOffset = get_mean_offset(segments, markers)
    # Make a copy for local use
    tmp_markers = deepcopy(markers)
    
    # Apply the offset
    for m in tmp_markers:
        m.start -= meanOffset
        if m.start < 0: m.start = 0
        
    # Allocate output matrix, give it alias mat for convenience.
    mat = ret['matrix'] = np.zeros((len(tmp_markers)-1, 12), dtype=np.float32)
    
    # Find the index of the segment that corresponds to the first marker
    f = lambda x: tmp_markers[0].start < x.start + x.duration
    index = (i for i,x in enumerate(segments) if f(x)).next()
    
    # Do the resampling
    try:
        for (i, m) in enumerate(tmp_markers):
            while segments[index].start + segments[index].duration < m.start + m.duration:
                dur = segments[index].duration
                if segments[index].start < m.start:
                    dur -= m.start - segments[index].start
                
                C = min(dur / m.duration, 1)
                
                mat[i, 0:12] += C * np.array(getattr(segments[index], feature))
                index += 1
            
            C = min( (m.duration + m.start - segments[index].start) / m.duration, 1)
            mat[i, 0:12] += C * np.array(getattr(segments[index], feature))
    except IndexError, e:
        pass # avoid breaking with index > len(segments)
    
    return ret


'''
###############################--EARWORM HELPER FUNCTIONS--##################################
Functionality taken from earworm/earworm_support.py
Used in Song.py and Mashup.py
'''

DEF_DUR = 600
MAX_SIZE = 800
MIN_RANGE = 16
MIN_JUMP = 16
MIN_ALIGN = 16
MAX_EDGES = 8
FADE_OUT = 3
RATE = 'beats'


## Print nodes and edges to screen, for debugging
def print_screen(graph):
    for n,d in graph.nodes_iter(data=True):
        print(n,d)
    for s,t,d in graph.edges_iter(data=True):
        print(s,t,d)

def analyze(track):
    timbre = resample_features(track, rate=RATE, feature='timbre')
    timbre['matrix'] = timbre_whiten(timbre['matrix'])
    pitch = resample_features(track, rate=RATE, feature='pitches')
    #NOTE why not whiten pitch matrix?
    
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
    
    markers = getattr(track.analysis, timbre['rate'])[timbre['index']:timbre['index']+len(paths)]
    graph = make_graph(paths, markers, timbre['matrix'], pitch['matrix'])
    
    #NOTE remove last node because empty?
    size = graph.number_of_nodes()
    graph.remove_node(size-1)
   
    return graph
 
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
        #NOTE Earworm alternate path interesections. Not necessary for labeling.
        #edges.extend([(markers[i].start, markers[l[0]+1].start, {'distance':l[1], 'duration': markers[i].duration, 'source':i, 'target':l[0]+1}) for l in paths[i]])
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
        DG.add_edge(source, target, 
                source=edge[0], target=edge[1],
                duration=edge[2]['duration'])
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
    distMat = np.sqrt(np.sum((repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2, axis=1, dtype=np.float32))
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

'''
###################################CAPSULE HELPER FUNCTIONS##############################
Functionality taken from capsule_support.py
Used in Mashup.py
'''

LOUDNESS_THRESH = -8

def equalize_tracks(tracks):
    
    def db_2_volume(loudness):
        return (1.0 - LOUDNESS_THRESH * (LOUDNESS_THRESH - loudness) / 100.0)
    
    for track in tracks:
        loudness = track.analysis.loudness
        track.gain = db_2_volume(loudness)
    
    return tracks
