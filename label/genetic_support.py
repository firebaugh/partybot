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
        self.max_source = 0
        self.min_source = sys.float_info.max
        for source in self.mashup.sources:
            if source.graph.number_of_nodes() < self.min_source:
                self.min_source = source.graph.number_of_nodes()
            if source.graph.number_of_nodes() > self.max_source:
                self.max_source = source.graph.number_of_nodes()
        self.sequence = sequence or self.genseq()
        self.fitness = None
        self.segs = None

    def transitions(self):
        indexes = []
        for i in range(0, self.length):
            if self.sequence[i] == 1:
                indexes.append(i)
        return indexes

    def segments(self):
        segs = []
        last = 0
        for i in range(1, self.length):
            if self.sequence[i] == 1 or i == self.length-1:
                segs.append( (last, i) )
                last = i+1
        return segs

    def genseq(self, n=0):
        #select enough transitions to spread source songs
        N = int(self.length/self.min_source) + 1
        if N < n: N = n
        sustain = False
        while sustain == False:
            sequence = [0] * (self.length)
            for i in range(N):
                sequence[random.randrange(1, self.length-2)] = 1
            #make sure some source song can align to the segments
            sustain = self.is_sustainable(sequence)
        return sequence

    def is_sustainable(self, sequence):
        last = 0
        for i in range(self.length):
            if sequence[i] == 1 or i == self.length-1:
                if (i-last) > self.max_source-1:
                    return False
                else:
                    last = i+1
        return True
    
    # Score individuals fitness
    # Store the best segments
    def _evaluate(self, optimum=None):
        #NOTE if no source songs match segment,
        #     re-randomize individual with >= # of transitions
        if self.is_sustainable(self.sequence) == False:
            self.sequence = self.genseq(len(self.transitions()))
        self.fitness, self.segs = self._score(self.sequence)
        return self.fitness

    # Score transitions based on possible mashup labeling
    def _score(self, sequence):
        tot_score = 0
        segs = {}
        # score each segment
        for seg in self.segments():
            mashup_seg = [self.mashup.mashup.graph.node[n] for n in range(seg[0], seg[1]+1)]
            leng = seg[1]-seg[0]
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
                        #segs[seg] = (source.mp3_name, score, song_seg)
                        segs[seg] = source.mp3_name
            tot_score = tot_score + min_score
        if tot_score == sys.float_info.max:
            print("?")
        return tot_score, segs

    #sigma scaling
    def _scale(self, mean, std):
        if std != 0:
            scaled_fitness = 1.0 + ( (self.fitness - mean)/(2.0*std) )
            if scaled_fitness < 0:
                return 0.1
            else:
                return scaled_fitness
        else:
            return 1.0
                
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
    # Use RANDOM operator:
    #   - Add a random transition
    #   - Delete a random transition
    #   - Move a random transition left or right
    def mutate(self):
        transitions = self.transitions()
        op = random.randrange(0,3)
        #NOTE if no transitions, add mutation
        if op == 0 or len(transitions) == 0:
            return self._add()
        elif op == 1:
            return self._delete(random.choice(transitions))
        else:
            return self._move(random.choice(transitions), random.choice(("l","r")))
       
    def _add(self):
        sequence = self.sequence
        indexes = []
        for i in range(len(sequence)):
            if sequence[i] == 0:
                indexes.append(i)
        sequence[random.choice(indexes)] = 1
        return sequence

    def _delete(self, transition):
        sequence = self.sequence
        sequence[transition] = 0
        return sequence

    def _move(self, transition, direction):
        sequence = self.sequence
        sequence[transition] = 0
        if direction == "l": #LEFT
            sequence[transition-1] = 1
        else: #RIGHT
            sequence[transition+1] = 1
        return sequence
    
    #TODO fix duplicated genes?
    def _repair(self, p0, p1):
        pass

    #TODO return mashup with individual's transitions
    # labeled with closest fit song segments
    def _finalize(self):
        pass

    def __repr__(self):
        return '<%s transitions="%s" fitness="%s">' % \
                (self.__class__.__name__, self.segs, self.fitness)

    # MINIMIZE score
    def __cmp__(self, other):
        return cmp(self.fitness, other.fitness)

    def copy(self):
        twin = self.__class__(self.mashup,self.sequence[:])
        twin.fitness = self.fitness
        return twin

class Environment(object):
    def __init__(self, mashup, population=None, size=100, maxgenerations=100,
            crossover_rate=0.90, mutation_rate=0.01, optimum=None):
        self.mashup = mashup
        self.size = size
        self.optimum = optimum
        self.population = population or self.genpop()
        print("Generated population.")
        for individual in self.population:
            individual._evaluate(self.optimum)
        print("Evaluated population.")
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
        self.population.sort()
        self.report()

    def _crossover(self):
        self._scale()
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
                individual._evaluate(self.optimum)
                next_population.append(individual)
        #NOTE: make pop bigger, then sort and cut
        for i in next_population:
            self.population.append(i)
        self.population.sort()
        self.population = self.population[:self.size]

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

    #sigma scaling
    def _scale(self):
        mean, std = self._evaluate()
        for individual in self.population:
            individual.fitness = individual._scale(mean, std)

    #mean and std of pop fitness
    def _evaluate(self):
        scores = [individual.fitness for individual in self.population]
        return np.mean(scores), np.std(scores)

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
        crossover_rate=0.9, mutation_rate=0.3):
    env = Environment(mashup, None, size, maxgenerations, crossover_rate, mutation_rate)
    env.run()

