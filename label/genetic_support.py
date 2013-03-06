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
    def _evaluate(self, cache):
        self.fitness, self.segs, cache = self._score(self.sequence, cache)
        return cache

    # Score transitions based on possible mashup labeling
    def _score(self, sequence, cache):
        tot_score = 0
        segs = {}
        # score each segment
        for seg in self.segments():
            leng = seg[1]-seg[0] #size of window
            min_score = sys.float_info.max
            
            # use min score of each source song
            for source in self.mashup.sources:
                beginnings = source.graph.number_of_nodes()-leng
                
                # sliding window distance
                for node in range(beginnings):
                
                    score = 0
                    c = 0
                    for n in range(node, node+leng+1):
                        # see if we have already calculated this
                        if (seg[0]+c, source.mp3_name, n) in cache:
                            score += cache[(seg[0]+c, source.mp3_name, n)]
                        # calculate and store new distance
                        else:
                            distance = feature_distance(
                                    self.mashup.mashup.graph.node[seg[0]+c], 
                                    source.graph.node[n])
                            cache[(seg[0]+c, source.mp3_name, n)] = distance
                            score += distance
                        c += 1
                    
                    # get min euclidean distance
                    score = np.sqrt(score)
                    if score < min_score:
                        min_score = score
                        #segs[seg] = (source.mp3_name, score, song_seg)
                        segs[seg] = source.mp3_name
            
            tot_score = tot_score + min_score
        return tot_score, segs, cache

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
        def mate(p0, p1):
            sequence = p0.sequence[:]
            sequence[left:right] = p1.sequence[left:right]
            child = p0.__class__(self.mashup,sequence)
            child._repair(p0,p1)
            return child
        # only returns sustainable offspring
        while True:
            left, right = self._pickpivots()
            child1, child2 = mate(self, other), mate(other, self)
            if self.is_sustainable(child1.sequence) and self.is_sustainable(child2.sequence):
                return child1, child2

    def _pickpivots(self):
        left = random.randrange(1, self.length-2)
        right = random.randrange(left, self.length-1)
        return left, right
    
    #fix duplicated sequences
    def _repair(self, p0, p1):
        sustain = False
        while sustain == False:
            # choice random: teleport or add
            op = random.choice([0,1])
            transitions = self.transitions()
            if op == 0:
                self.sequence[random.choice(transitions)] = 0
            self.sequence[random.randrange(self.length)] = 1
            # ensure sustainability
            sustain = self.is_sustainable(self.sequence)

    # Pick a uniformly random transition to mutate
    # Use OPTIMAL operator:
    #   - Add a random transition
    #   - Delete a random transition
    #   - Move a random transition left or right
    def mutate(self, cache):
        transition = random.choice(self.transitions())
        # only consider sustainable mutations
        mutations = []
        add, cache = self._add(cache)
        if self.is_sustainable(add[1]) == True:
            mutations.append(add)
        delete, cache = self._delete(transition, cache)
        if self.is_sustainable(delete[1]) == True:
            mutations.append(delete)
        move, cache = self._move(transition, random.choice( ("l","r") ), cache)
        if self.is_sustainable(move[1]) == True:
            mutations.append(move)
        # use optimal mutation
        self.fitness, self.sequence, self.segs = max(mutations)
        return cache
       
    def _add(self, cache):
        sequence = self.sequence
        indexes = []
        for i in range(len(sequence)):
            if sequence[i] == 0:
                indexes.append(i)
        sequence[random.choice(indexes)] = 1
        fitness, segs, cache = self._score(sequence, cache)
        return (fitness, sequence, segs), cache

    def _delete(self, transition, cache):
        sequence = self.sequence
        sequence[transition] = 0
        fitness, segs, cache = self._score(sequence, cache)        
        return (fitness, sequence, segs), cache

    def _move(self, transition, direction, cache):
        sequence = self.sequence
        sequence[transition] = 0
        if direction == "l": #LEFT
            sequence[transition-1] = 1
        else: #RIGHT
            sequence[transition+1] = 1
        fitness, segs, cache = self._score(sequence, cache)        
        return (fitness, sequence, segs), cache

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
            crossover_rate=0.90, mutation_rate=0.01, optimum=0.0):
        self.mashup = mashup
        self.size = size
        self.optimum = optimum
        self.population = population or self.genpop()
        self.cache = {} #{(mashup_node, source_name, source_node): feature_distance}
        for individual in self.population:
            self.cache = individual._evaluate(self.cache)
        self._scale()
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
        return self.generation > self.maxgenerations or self.best.fitness == self.optimum

    def step(self):
        self._scale()
        self.population.sort()
        self._crossover()
        self.generation += 1
        self.population.sort()
        self.report()

    def _crossover(self):
        next_population = [self.best.copy()]
        mates = self._select() #select possible parents using SUS with sigma scaling
        while len(next_population) < self.size:
            mate1 = random.choice(mates)
            if random.random() < self.crossover_rate:
                mate2 = random.choice(mates) #TODO is this right?
                offspring = mate1.crossover(mate2)
            else:
                offspring = [mate1.copy()]
            for individual in offspring:
                self._mutate(individual)
                individual._evaluate(self.cache)
                next_population.append(individual)
        # add new pop to old pop, sort, then cut off at pop max size
        for i in next_population:
            self.population.append(i)
        self.population.sort()
        self.population = self.population[:self.size]

    def _select(self):
        return self.SUS()

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:
            self.cache = individual.mutate(self.cache)

    # stochastic universal sampling
    def SUS(self):
        mates = []
        ptr = random.random()
        accum = 0
        for i in range(self.size):
            accum += self.population[i].fitness
            if accum > ptr:
                mates.append(self.population[i])
                ptr += 1
        return mates

    def _scale(self):
        #sigma scaling
        mean, std = self._evaluate()
        accum = 0
        for individual in self.population:
            individual.fitness = individual._scale(mean, std)
            accum += individual.fitness
        #normalize
        for individual in self.population:
            individual.fitness /= accum

    def _evaluate(self):
        scores = [individual.fitness for individual in self.population]
        return np.mean(scores), np.std(scores)


    def report(self):
        print("="*70)
        print("generation: %s" % self.generation)
        print("best: %s" % self.best)

    def _finalize(self):
        return self.best._finalize()

# Sum feature distance between two nodes
# Nodes have two feature vectors, pitch and timbre, each of 12 floats
def feature_distance(n1, n2):
    distance = 0
    for f in range(12):
        distance += pow((n1['pitch'][f] - n2['pitch'][f]),2)
        distance += pow((n1['timbre'][f] -n2['timbre'][f]),2)
    return distance


def genetic_labeling(mashup, size=300, maxgenerations=100, 
        crossover_rate=0.9, mutation_rate=0.3, optimum=0.0):
    env = Environment(mashup, None, size, maxgenerations, crossover_rate, mutation_rate, optimum)
    env.run()

