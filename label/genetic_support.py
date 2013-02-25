"""
genetic_support.py

Shell for genetic algorithm. Applied to mashup in Label.py

Created by Caitlyn Clabaugh
Based on: http://code.activestate.com/recipes/199121-a-simple-genetic-algorithm/
"""

import sys
import numpy as np
import random

class Individual(object):
    
    def __init__(self, mashup, sequence=None):
        self.mashup = mashup
        self.length = mashup.mashup.graph.number_of_nodes()
        self.sequence = sequence or self.genseq()
        self.score = None

    def transitions(self):
        indexes = []
        for i in range(0,len(self.sequence)):
            if self.sequence[i] == 1:
                indexes.append(i)
        return indexes

    def segments(self):
        segs = []
        last = 0
        for i in self.transitions():
            segs.append( (last, i) )
            last = i+1
        if self.length > last: segs.append( (last, (self.length-1)) )
        return segs

    # Generate sequence 2 randomly placed transitions (1s)
    def genseq(self, n=2):
        sequence = [0] * (self.length-1)
        for i in range(n):
            sequence[random.randrange(0, len(sequence))] = 1
        return sequence
    
    # Evaluate individual using scoring function
    def evaluate(self, optimum=None):
        self.score = self._score(self.sequence)
        return self.score

    # Score transitions based on possible mashup labeling
    def _score(self, sequence):
        tot_score = 0
        # score each segment
        for seg in self.segments():
            mashup_seg = [self.mashup.mashup.graph.node[n] for n in range(seg[0],seg[1]+1)]
            leng = seg[1]-seg[0]
            #NOTE if no source songs match segment (too long), then max float is cost?
            min_score = sys.float_info.max
            # use min score of each source song
            for source in self.mashup.sources:
                beginnings = source.graph.number_of_nodes()-leng
                # sliding window distance
                for node in range(beginnings):
                    song_seg = [source.graph.node[n] for n in range(node, node+leng+1)]
                    # get distance between source segment and mashup segment features
                    score = feature_distance(mashup_seg, song_seg)
                    if score < min_score:
                        min_score = score
            tot_score = tot_score + min_score
        return tot_score
                
    #NOTE is two-point cross over the best?
    def crossover(self, other):
        return self._twopoint(other)

    def _twopoint(self, other):
        left, right = self._pickpivots()
        def mate(p0, p1):
            sequence = p0.sequence[:]
            sequence[left:right] = p1.sequence[left:right]
            child = p0.__class__(self.mashup,sequence)
            child._repair(p0,p1)
            return child
        return mate(self, other), mate(other, self)

    def _pickpivots(self):
        left = random.randrange(1, self.length-2)
        right = random.randrange(left, self.length-1)
        return left, right

    # Pick a uniformly random transition to mutate
    # Use random (NOTE use max scoring operator? expensive!) operator:
    #   - Add a random transition
    #   - Delete a random transition
    #   - Move a random transition left or right
    def mutate(self):
        transitions = self.transitions()
        #op = random.randrange(0,3)
        #NOTE is no transitions in segment, use add mutation
        #if op == 0 or len(transitions) == 0:
        #    return self._add()
        #elif op == 1:
        #    return self._delete(random.choice(transitions))
        #else:
        #    return self._move(random.choice(transitions), random.choice(("l","r")))
        if len(transitions) == 0:
            add = self._add()
            return add[1]
        else:
            transition = random.choice(transitions)
            add = self._add()
            delete = self._delete(transition)
            move = self._move(transition, random.choice("l","r"))
            max_op = max(add, delete, move)
            return max_op[1]

    def _add(self):
        sequence = self.sequence
        indexes = []
        for i in range(len(sequence)):
            if sequence[i] == 0:
                indexes.append(i)
        sequence[random.choice(indexes)] = 1
        #return sequence
        return (self._score(sequence), sequence)

    def _delete(self, transition):
        sequence = self.sequence
        sequence[transition] = 0
        #return sequence
        return (self._score(sequence), sequence)

    def _move(self, transition, direction):
        sequence = self.sequence
        sequence[transition] = 0
        if direction == "l": #LEFT
            sequence[transition-1] = 1
        else: #RIGHT
            sequence[transition+1] = 1
        #return sequence
        return (self._score(sequence), sequence)
    
    #TODO fix duplicated genes?
    def _repair(self, p0, p1):
        pass

    #TODO return mashup with individual's transitions
    # labeled with closest fit song segments
    def _finalize(self):
        pass

    def __repr__(self):
        return '<%s transitions="%s" score="%s">' % \
                (self.__class__.__name__, self.transitions(), self.score)

    # MINIMIZE score
    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def copy(self):
        twin = self.__class__(self.mashup,self.sequence[:])
        twin.score = self.score
        return twin

class Environment(object):
    def __init__(self, mashup, population=None, size=100, maxgenerations=100,
            crossover_rate=0.90, mutation_rate=0.01, optimum=None):
        self.mashup = mashup
        self.size = size
        self.optimum = optimum
        self.population = population or self.genpop()
        for individual in self.population:
            individual.evaluate(self.optimum)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.maxgenerations = maxgenerations
        self.generation = 0
        self.report()

    def best():
        def fget(self):
            return self.population[0]
        return locals()
    best = property(**best())

    def genpop(self):
        return [Individual(self.mashup) for individual in range(self.size)]

    def run(self):
        while not self._goal():
            self.step()
        return self._finalize()

    def _goal(self):
        return self.generation > self.maxgenerations

    def step(self):
        self.population.sort()
        self._crossover()
        self.generation += 1
        self.report()

    def _crossover(self):
        next_population = [self.best.copy()]
        while len(next_population) < self.size:
            mate1 = self._select()
            if random.random() < self.crossover_rate:
                mate2 = self._select()
                offspring = mate1.crossover(mate2)
            else:
                offspring = [mate1.copy()]
            for individual in offspring:
                self._mutate(individual)
                individual.evaluate(self.optimum)
                next_population.append(individual)
            self.population = next_population[:self.size]

    #NOTE is this a good selection method?
    def _select(self):
        return self._tournament()

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:
            individual.mutate()

    def _tournament(self, size=8, choosebest=0.90):
        competitors = [random.choice(self.population) for i in range(size)]
        competitors.sort()
        if random.random()< choosebest:
            return competitors[0]
        else:
            return random.choice(competitors[1:])

    def report(self):
        print("="*70)
        print("generation: %s" % self.generation)
        print("best: %s" % self.best)

    def _finalize(self):
        return self.best._finalize()

# Euclidean distance between two lists of nodes
# Nodes have two feature vectors, pitch and timbre, each of 12 floats
def feature_distance(seg1, seg2):
    distance = 0
    for n in range(len(seg1)):
        for f in range(12):
            distance = distance + pow((seg1[n]['pitch'][f] - seg2[n]['pitch'][f]),2)
            distance = distance + pow((seg1[n]['timbre'][f] - seg2[n]['timbre'][f]),2)
    return np.sqrt(distance)


def genetic_labeling(mashup, size=100, maxgenerations=100, 
        crossover_rate=0.9, mutation_rate=0.2):
    env = Environment(mashup, None, size, maxgenerations, crossover_rate, mutation_rate)
    env.run()

