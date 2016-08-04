# Currently in desperate need of a revamp
#  - Need better error function for judging fitness; sheer magnitude of wrong answers vs. right answers is very large

from __future__ import division
from deps.peas.networks.rnn import NeuralNetwork
from deps.peas.methods.neat import NEATPopulation, NEATGenotype
from itertools import repeat, starmap, chain
from collections import namedtuple
import numpy as np
import random
import cPickle
import csv

def unravel(n, min_length):
    output = []
    while n > 0:
        output.append(float(n & 1))
        n = n >> 1
    padding = []
    while len(output) + len(padding) < min_length:
        padding.append(0)
    return output + padding

def pack(a):
    b = 1
    s = 0
    for bit in a:
        if bit > 0:
            s += b
        b += b
    return s

def fitnessFunction(output,target):
    assert len(output) == len(target)
    indices = [i for i in range(len(target))]
    indices.reverse()
    s = 0
    for o in indices:
        if output[o] > 0.5:
            bit = 1
        else:
            bit = 0
        if bit == target[0]:
            s += 1
        else:
            return s
    return s

class MultiplicationTask(object):
    def __init__(self, run_length, bit_sizes, accuracy_threshold, runs_file):
        #Create fixed runs
        inputs = [(random.getrandbits(bit_sizes), random.getrandbits(bit_sizes)) for i in range(run_length)]
        # self.inputs = [tuple(chain([unravel(n, bit_sizes) for n in item])) for item in inputs]
        self.inputs = np.array([tuple(unravel(x, bit_sizes) + unravel(y, bit_sizes)) for x, y in inputs], dtype=float)
        self.outputs = [unravel(a*b,bit_sizes*2) for a, b in inputs]
        self.pairs = zip(self.inputs, self.outputs)
        for t, product in zip(inputs, [a*b for a, b in inputs]):
            x, y = t
            runs_file.write(','.join(['fixed', str(x), str(y), str(product)]))
            runs_file.write('\n')
        self.bit_sizes = bit_sizes
        self.run_length = run_length
        self.min_accuracy = accuracy_threshold
        self.runs_file = runs_file

    def evaluate(self, network):
        if not isinstance(network,NeuralNetwork):
            network = NeuralNetwork(network)
        network.make_feedforward()
        score = 0
        for (i, target) in self.pairs:
            output = network.feed(i)
            output = output[-self.bit_sizes*2:]
            score += fitnessFunction(output,target)
        return {'fitness': score/len(self.pairs)}

    def solve(self, network):
        """ Defines if a network has solved the problem. Size matters more than accuracy, but error should not exceed a certain margin. """
        return self.evaluate(network)['fitness'] >= min_accuracy

TaskParams = namedtuple('TaskParams', ['bitsize', 'weight_range', 'neuron_type', 'popsize', 'run_length', 'accuracy_threshold', 'generations', 'ann_file', 'runs_file'])

def convertParameters(bitsize, weight_min, weight_max, neuron_type, popsize, run_length, accuracy_threshold, generations, ann_file, runs_file):
    return TaskParams(bitsize=int(bitsize),
                          weight_range=tuple((float(weight_min),float(weight_max))),
                          neuron_type=neuron_type,
                          popsize=int(popsize),
                          run_length=int(run_length),
                          accuracy_threshold=float(accuracy_threshold),
                          generations=int(generations),
                          ann_file=ann_file,
                          runs_file=runs_file)

with open("example.csv", "r") as ps:
    for p in starmap(convertParameters, csv.reader(ps)):
        def genotype():
            i = 0
            inputs = [r+1 for r in range(p.bitsize*2)]
            outputs = [p.bitsize*2+a for a in range(p.bitsize*2)]
            l = random.randrange(0, p.bitsize*2+1)
            assert l <= p.bitsize*2
            i = random.sample(l,inputs)
            o = random.sample(l,outputs)
            random.shuffle(i)
            random.shuffle(o)
            return NEATGenotype(inputs=p.bitsize*2,
                                outputs=p.bitsize*2,
                                topology=(inputs, outputs),
                                weight_range=p.weight_range,
                                types=[p.neuron_type],
                                stop_when_solved=True)
        with open(p.runs_file, "w") as r:
            r.write('type,x,y,product')
            pop = NEATPopulation(genotype, popsize=p.popsize)
            task = MultiplicationTask(p.run_length, p.bitsize, p.accuracy_threshold, r)
            pop.epoch(generations=p.generations, evaluator=task, solution=task) #Run the task code
            with open(p.ann_file, "wb") as a:
                cPickle.dump(pop, a, -1) #Serialize the population to a file
