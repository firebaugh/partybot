#/usr/bin/env python
# encoding: utf=8

"""
Song.py

Accepts a mp3 and creates a graph of timbre and pitch features using Echo Nest.
Provides functions to print graph to screen, draw graph as .png, create an "earworm" graph,
render full track to .mp3

Created by Caitlyn Clabaugh
Based on:
  - earworm.py by Tristan Jehan and Jason Sundram
"""

#hard coded, need to change for each developer :/
from pyechonest import config
config.ECHO_NEST_API_KEY = "TTAPZNVYMGG5KQBJI"

from optparse import OptionParser
import sys
from string import lower, split

try:
    import networkx as nx
except ImportError:
    print """Song.py requires networkx. 
    
If setuptools is installed on your system, simply:
easy_install networkx 

Otherwise, you can get it here: http://pypi.python.org/pypi/networkx

Get the source, unzip it, cd to the directory it is in and run:
    python setup.py install
"""
    sys.exit(1)

from echonest.action import Playback, render
from echonest.audio import LocalAudioFile
# from echonest.cloud_support import AnalyzedAudioFile
from earworm_support import analyze

class Song:

    def __init__(self, mp3_filename, recompute = False, verbose = False):
        self.mp3_filename = mp3_filename
        self.mp3_path = split(lower(self.mp3_filename), '.mp3')[0]
        self.mp3_name = split(self.mp3_path, '/').pop()
        self.graph_filename = self.mp3_path+".graph"
        self.graph = None
        self.track = None # Echo Nest track
        self.run(recompute, verbose)

    def run(self, recompute = False, verbose = False): 
        if verbose:  print("Probing for pre-existing graph %s..." % self.graph_filename)
        if self.load_graph(self.graph_filename) is False or recompute is True:
            if verbose: print("Computing graph for track %s..." % self.mp3_path)
            self.graph = self.compute_graph(verbose)
            if verbose: print("Writing computed graph out to file %s..." % self.graph_filename)
            self.write_graph(self.graph_filename)
        if verbose: print("Created song graph.")
        return self.graph

    def load_graph(self, graph_filename):
        try:
            f = open(graph_filename, "r")
            lines = f.readlines()
            f.close()

            # num node lines, num edge lines
            # TODO track analysis data
            n,e = lines[0].split(" ")
            n = (int(n)*3+1)
            e = int(e) + n
            # NODES: (node_index, {'timbre', 'pitch'})
            nodes = []
            for i in range(1,n,3):
                features = {}
                timbre = lines[i+1].split(" ")
                pitch = lines[i+2].split(" ")
                features['timbre'] = [float(j) for j in timbre]
                features['pitch'] = [float(k) for k in pitch]
                nodes.append( (int(lines[i]),features) )
            # EDGES: (source_node_index, target_node_index
            #           {'duration','source','target'})
            edges = []
            for j in range(n+1,e,2):
                e = lines[j].split(" ")
                edges.append( (int(e[0]), int(e[1]), {'duration': float(e[2]), 
                    'source': float(e[3]), 'target': float(e[4])}) ) 
            # make digraph
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)
            return self.graph
        except:
            # No pre-existing graph
            return False

    # Load audio info from EchoNest
    def load_track(self, verbose = False):
        self.track = LocalAudioFile(self.mp3_filename, verbose)
        return self.track

    def compute_graph(self, verbose = False):
        # Load audio info from EchoNest
        self.track = self.load_track(verbose)
        self.graph = analyze(self.track)
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
        for s,t,d in self.graph.edges_iter(data=True):
            f.write(" ".join([str(s),str(t),
                str(d['duration']),str(d['source']),str(d['target'])]))
            f.write("\n")
        f.close()

    def render(self, mp3_filename):
        # Check that we have loaded track from Echo Nest
        if self.track == None:
            self.load_track(True)

        # NOTE to shorten/lengthen refer to compute_path() in earworm.py
        # renders full length of song
        actions = [Playback(self.track, min(self.graph.nodes()), max(self.graph.nodes()))]
        render(actions, mp3_filename) 
        
    def __print__(self):
        for n,d in self.graph.nodes_iter(data=True):
            print(n,d)


def main():
    # Command line options
    usage = "usage: %s [options] <path_to_mp3>" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    parser.add_option("-f", "--force", action="store_true", help="force recompute graph")
    parser.add_option("-r", "--render", action="store_true", help="render song graph to mp3.")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        parser.print_help()
        return -1
    recompute = options.force
    verbose = options.verbose
    render = options.render

    song = Song(args[0], recompute, verbose)
    if render: song.render(song.mp3_path+"-out.mp3")

    return 1

if __name__ == "__main__":
    main()
