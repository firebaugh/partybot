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
    
    def __init__(self, mashup, restrict=0, sequence=None):
        self.mashup = mashup
        self.length = mashup.mashup.graph.number_of_nodes()
        self.max_source = 0
        self.min_source = sys.float_info.max
        for source in self.mashup.sources:
            if source.graph.number_of_nodes() < self.min_source:
                self.min_source = source.graph.number_of_nodes()
            if source.graph.number_of_nodes() > self.max_source:
                self.max_source = source.graph.number_of_nodes()
        self.restrict = restrict
        self.sequence = sequence or self.genseq(self.restrict)
        self.fitness = None
        self.rank = None
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
        if n < N: n = N
        sustain = False
        while sustain == False:
            sequence = [0] * (self.length)
            for i in range(n):
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
        fitness, self.segs, cache = self._score(self.sequence, cache)
        #NOTE closer score is to 0, the better
        self.fitness = -1 * fitness
        return cache

    # Score transitions based on possible mashup labeling
    def _score(self, sequence, cache):
        tot_score = 0
        segs = {}
        # score each segment
        for seg in self.segments():
            
            # use min score of each source song
            leng = seg[1]-seg[0] #size of window
            min_score = sys.float_info.max
            for source in self.mashup.sources:
                
                # sliding window distance
                beginnings = source.graph.number_of_nodes()-leng
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
                    if score < min_score:
                        min_score = score
                        #{ (mashup_start, mashup_stop): (source_name, (source_start, source_stop)), ... }
                        segs[seg] = (source.mp3_name, (node, node+leng))
            
            tot_score += min_score
        return np.sqrt(tot_score), segs, cache

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
            child = p0.__class__(self.mashup,self.restrict,sequence)
            child._repair()
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
    
    # restrict # of transitions by self.restrict
    # NOTE need restrict to be sustainable; otherwise, run forever!
    def _repair(self):
        trans = self.transitions()
        diff = len(trans) - self.restrict
        sequence = self.sequence
        # too many
        if diff > 0:
            for d in range(diff):
                rand = random.choice(trans)
                sequence[rand] = 0
                while self.is_sustainable(sequence) == False:
                    sequence[rand] = 1 # replace
                    rand = rand.choice(trans) # choose another
                    sequence[rand] = 0
        #too few
        elif diff < 0:
            cont = [i for i in range(len(self.sequence))]
            for t in trans: cont.remove(t)
            for d in range(-1*diff):
                rand = random.choice(cont)
                sequence[rand] = 1

        self.sequence = sequence
        return sequence

    # Pick a uniformly random transition to mutate
    # Add will always be sustainable
    # Use OPTIMAL operator:
    #   - Add a random transition
    #   - Delete a random transition
    #   - Move a random transition left or right
    def mutate(self, cache):
        transitions = self.transitions()
        if len(transitions) < 1:
            add, cache = self._add(cache)
            self.fitness, self.sequence, self.segs = add
            return cache
        else:
            transition = random.choice(transitions)
            add, cache = self._add(cache)
            delete, cache = self._delete(transition, cache)
            move, cache = self._move(transition, random.choice( ("l","r") ), cache)
            # use optimal mutation
            self.fitness, self.sequence, self.segs = max(add, delete, move)
            return cache
       
    def _add(self, cache):
        #add to non transition locations
        indexes = []
        for i in range(len(self.sequence)):
            if self.sequence[i] == 0:
                indexes.append(i)
        #ensure sustainability
        sustain = False
        while sustain == False:
            sequence = self.sequence
            sequence[random.choice(indexes)] = 1
            sustain = self.is_sustainable(sequence)
        fitness, segs, cache = self._score(sequence, cache)
        return (fitness, sequence, segs), cache

    def _delete(self, transition, cache):
        random.choice(self.transitions())
        sequence = self.sequence
        sequence[transition] = 0
        fitness, segs, cache = self._score(sequence, cache)        
        return (fitness, sequence, segs), cache

    def _move(self, transition, direction, cache):
        sequence = self.sequence
        sequence[transition] = 0
        if direction == "l": #LEFT
            if transition > 1:
                sequence[transition-1] = 1
            else: #wrap
                sequence[len(sequence)-2] = 1
        else: #RIGHT
            if transition < len(sequence)-3:
                sequence[transition+1] = 1
            else: #wrap
                sequence[1] = 1
        fitness, segs, cache = self._score(sequence, cache)        
        return (fitness, sequence, segs), cache

    # Label mashup graph with closest fit song segments
    def _finalize(self, cache, verbose):
        # {mashup_node: {timbre: [], pitch: [], label: (song_mp3_name, song_node, segment_number)}, ...}
        if verbose: print("Finalizing...")
        labeled = self.mashup.mashup.graph.to_directed()
        self._evaluate(cache)
      
        s = 0
        for t in self.segs.keys():
            label = self.segs[t][0] #song name
            seg = self.segs[t][1] #song segment
            j = 0
            for i in range(t[0], t[1]+1):
                labeled.node[i]['label'] = (label, seg[0]+j, s)
                j += 1
            s += 1

        return labeled

    def __repr__(self):
        return '<%s transitions="%s" fitness="%s">' % \
                (self.__class__.__name__, self.segs, self.fitness)

    # MAXIMIZE score
    def __cmp__(self, other):
        return cmp(other.fitness, self.fitness)

    def copy(self):
        twin = self.__class__(self.mashup,self.restrict,self.sequence[:])
        twin.fitness = self.fitness
        return twin

alpha = 0.2

class Environment(object):
    def __init__(self, mashup, population=None, size=100, maxgenerations=100,
            crossover_rate=0.90, mutation_rate=0.01, optimum=0.0,
            restrict=0, converge=True, smooth=False, verbose=False, plot=None):
        #env and pop setup
        self.mashup = mashup
        self.size = size
        self.optimum = optimum
        self.restrict = restrict
        self.converge = converge
        self.smooth = smooth
        self.population = population or self.genpop() #NOTE use restricted unless too small
        self.cache = {} #{(mashup_node, source_name, source_node): feature_distance}
        for individual in self.population:
            self.cache = individual._evaluate(self.cache)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.maxgenerations = maxgenerations
        self.generation = 0
        self.past_average = 0
        self.curr_average = self.get_average()
        self.past_exp_average = 0
        self.curr_exp_average = self.get_exp_average()
        #user commands for verbose and plot
        self.verbose = verbose
        if plot: self.plot = plot+".dat"
        else: self.plot = False
        if self.plot:
            f = open(self.plot, "w")
            f.write(
'''# GA for labeling mashup %s
# size = %d
# crossover rate = %f
# mutation rate = %f
# optimum = %f
# ------------------------------------------
# GEN BEST_FITNESS AVG_FITNESS WEIGHTED_AVG
'''
                    % (self.mashup.mashup.mp3_name, self.size,
                        self.crossover_rate, self.mutation_rate, self.optimum))
            f.close()
        if self.verbose: self.report()

    def best():
        def fget(self):
            return self.population[0]
        return locals()
    best = property(**best())

    def get_average(self):
        return np.mean([individual.fitness for individual in self.population])

    def get_exp_average(self):
        return alpha*self.past_average + (1-alpha)*self.curr_average 

    def genpop(self):
        return [Individual(self.mashup,self.restrict) for individual in range(self.size)]

    def run(self):
        while not self._goal():
            self.step()
        return self._finalize()

    def _goal(self):
        if self.converge:
            if self.smooth: return abs(self.past_exp_average - self.curr_exp_average) <= self.optimum
            else: return abs(self.past_average - self.curr_average) <= self.optimum
        else:
            return self.generation >= self.maxgenerations

    def step(self):
        self._crossover()
        # update vars
        self.generation += 1
        self.past_average = self.curr_average
        self.curr_average = self.get_average()
        self.past_exp_average = self.curr_exp_average
        self.curr_exp_average = self.get_exp_average()
        # report results
        if self.verbose: self.report()
        if self.plot: self._plot()

    def _crossover(self):
        mates = self._select()
        next_population = [self.best.copy()]
        while len(next_population) < self.size:
            index = random.randrange(len(mates))
            mate1 = mates[index]
            #crossover
            if random.random() < self.crossover_rate:
                mate2 = random.choice(mates) 
                offspring = mate1.crossover(mate2)
                if self.restrict: #NOTE restricting transitions
                    for individual in offspring:
                        individual._repair()
            else:
                offspring = [mate1.copy()]
            #mutate
            for individual in offspring:
                if self.restrict: self.restricted_mutate(individual) #NOTE restricting transitions (just move operator)
                else: self._mutate(individual)
                next_population.append(individual)
        
        # add new pop to old pop, sort, then cut off at pop max size
        self.population += next_population
        for individual in self.population:
            self.cache = individual._evaluate(self.cache)
        self.population.sort()
        self.population = self.population[:self.size]

    def _select(self):
        self._scale()
        self.population.sort()
        return self.weighted_choice()

    #weighted random selection by fitness
    def weighted_choice(self):
        mates = []
        weights =  [i.fitness for i in self.population]
        rnd = random.random() * sum(weights) 
        for i,w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                mates.append(self.population[i])
        return mates

    #select top n individuals
    def top(self, n=0):
        self.population.sort()
        return self.population[:n]
 
    #rank by fitness
    def _rank(self):
        self.population.sort()
        r = 1
        for individual in self.population:
            individual.rank = r
            r += 1

    #sigma scaling
    def _scale(self):
        mean, std = self._evaluate()
        for individual in self.population:
            individual.fitness = individual._scale(mean, std)
        
    #normalize fitness between 0 and 1
    def _normalize(self):
        accum = sum([individual.fitness for individual in self.population])
        for individual in self.population:
            individual.fitness /= accum
   
   #evaluate population mean and std
    def _evaluate(self):
        scores = [individual.fitness for individual in self.population]
        return np.mean(scores), np.std(scores)

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:
            self.cache = individual.mutate(self.cache)

    # mutate with move operation only; for restricted crossover
    def restricted_mutate(self, individual):
        if random.random() < self.mutation_rate:
            transition = random.choice(individual.transitions())
            direction= random.choice( ("l","r") )
            move, self.cache = individual._move(transition, direction, self.cache)
            individual.fitness, individual.sequence, individual.segs = move

    #randomize the population
    #leave best
    #use same # of transitions
    def _randomize(self, population):
       population.sort()
       new_population = [population[0]]
       for i in range(len(population)-1):
           if random.random() > 0.5:
               population[i].sequence = population[i].genseq()
           new_population.append(population[i])
       return new_population

    def report(self):
        print("="*70)
        print("generation: %s" % self.generation)
        print("size: %s" % self.size)
        print("mean: %s \n std: %s" % (self._evaluate()))
        print("ewa: %s" % self.curr_exp_average)
        print("best: %s" % self.best)

    def _plot(self):
        f = open(self.plot, "a")
        f.write("%d\t%f\t%f\t%f\n" % (self.generation, self.best.fitness, self.curr_average, self.curr_exp_average))
        f.close()
    
    def _finalize(self):
        return self.best._finalize(self.cache, self.verbose)

# Sum feature distance between two nodes
# Nodes have two feature vectors, pitch and timbre, each of 12 floats
def feature_distance(n1, n2):
    distance = 0
    for f in range(12):
        distance += pow((n1['pitch'][f] - n2['pitch'][f]),2)
        distance += pow((n1['timbre'][f] -n2['timbre'][f]),2)
    return distance


def genetic_labeling(mashup, verbose=False, out=None,
        size=300, maxgenerations=10, crossover_rate=0.9, mutation_rate=0.2, optimum=0.0,
        restrict=0, converge=True, smooth=False):
    env = Environment(mashup, None, size, maxgenerations, 
            crossover_rate, mutation_rate, optimum,
            restrict, converge, smooth, verbose, out)
    return env.run()

