"""
alignment_support.py

Supports the alignment between a mashup and its N source songs.
Matches sites using iterative Smith-Waterman local sequence alignment.

Created by Caitlyn Clabaugh
Based on:
  - alignment.py by Aleksandr Levchuk
"""

#hard coded, need to change for each developer :/
from pyechonest import config
config.ECHO_NEST_API_KEY = "TTAPZNVYMGG5KQBJI"

import numpy as np

## Iterative local sequence alignment (smith-waterman)
## Returns labeled graph of mashup using source songs
## mashup = type Mashup
def align(mashup, verbose=False):
    mashup_graph = mashup.mashup.graph
    unlabeled = mashup_graph.nodes()

    # Loop until whole graph completely labeled
    while(len(unlabeled) > 0):
        max_score = 0
        song, label, unlabel = None, None, None
        
        for s in mashup.sources:
            tmp_label, tmp_unlabel, score = water(mashup_graph, unlabeled, s.graph, s.mp3_name)
            # Use best score between soruces
            if score > max_score:
                song = s.mp3_name
                label = tmp_label
                unlabel = tmp_unlabel
                max_score = score

        mashup_graph = label.to_directed()
        unlabeled = unlabel
        if verbose: print("Found %s segment. Remaining: %d" % (song, len(unlabeled)))
    
    return mashup_graph

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
    return np.sqrt(dist)

## Standard deviation function
def standardDev(dist_list):
    mean = sum(dist_list)/len(dist_list)
    mean_sq_dist = [ pow((x - mean), 2) for x in dist_list ]    
    return np.sqrt(sum(mean_sq_dist)/len(dist_list))
 


