from optparse import OptionParser
import sys

try:
    import networkx as nx
except ImportError:
    print """training_data.py requires networkx. 
"""
    sys.exit(1)

def load_graph(node_file, edge_file, verbose=False):
    # node file
    try:
        f = open(node_file, "r")
        node_lines = f.readlines()
        f.close()
    except:
        print("Node file %s not found." % node_file)
        exit(0)
    # edge file
    try:
        f = open(edge_file, "r")
        edge_lines = f.readlines()
        f.close()
    except:
        print("Edge file %s not found." % edge_file)
        exit(0)

    # load nodes
    nodes = []
    for i in range(len(node_lines)):
        n = node_lines[i].split(" ")
        nodes.append( (i+1, {'song': n[1], 'in-out': (int(n[2]), int(n[3])), 
            'duration': int(n[3])-int(n[2]), 'features': [float(a) for a in n[4:]]}) )
    if verbose: print("Loaded nodes.")

    # load edges
    edges = []
    for j in range(len(edge_lines)):
        e = edge_lines[j].split(" ")
        edges.append( (int(e[0]), int(e[1])) )
    if verbose: print("Loaded edges.")

    # make graph
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if verbose: print("Created graph.")
    return graph


def write_data(graph, out_file):
    f = open(out_file, "w")
    for n1,d1 in graph.nodes_iter(data=True):
        for n2,d2 in graph.nodes_iter(data=True):
            #features
            for i in d1['features']:
                f.write("%.16f," % i)
            for j in d2['features']:
                f.write("%.16f," % j)
            #label
            if graph.has_edge(n1, n2): f.write("true\n")
            else: f.write("false\n")
    f.close()

def main():
    # Command line options
    usage = "usage: %s [options] <node_file> <edge_file> <out_file>" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="show results on screen")
    (options, args) = parser.parse_args()
    if len(args) < 3:
        parser.print_help()
        return -1
    verbose = options.verbose

    if verbose: print("Loading graph from %s and %s..." % (args[0], args[1]))
    graph = load_graph(args[0], args[1], verbose)

    if verbose: print("Writing feature data to %s..." % args[2])
    write_data(graph, args[2])

    return 1

if __name__ == "__main__":
    main()
