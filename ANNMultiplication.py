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

class MultiplicationTask(object):
    def __init__(self, run_length, bit_sizes, accuracy_threshold, runs_file):
        #Create fixed runs
        inputs = [(random.getrandbits(bit_sizes), random.getrandbits(bit_sizes)) for i in range(run_length)]
        # self.inputs = [tuple(chain([unravel(n, bit_sizes) for n in item])) for item in inputs]
        self.inputs = [tuple(unravel(x, bit_sizes) + unravel(y, bit_sizes)) for x, y in inputs]
        self.outputs = [a*b for a, b in inputs]
        for t, product in zip(inputs, self.outputs):
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
        if not network.node_types[-1](-1000) < -0.95:
            raise Exception("Network should be able to output value of -1, e.g. using a tanh node.")
        #Create random runs (so as to prevent overfitting)
        new_inputs = [(random.getrandbits(self.bit_sizes), random.getrandbits(self.bit_sizes)) for i in range(self.run_length)]
        new_outputs = [a*b for a, b in new_inputs]
        for t, product in zip(new_inputs, new_outputs):
            x, y = t
            self.runs_file.write(','.join(['random', str(x), str(y), str(product)]))
            self.runs_file.write('\n')
        inputs = self.inputs + [tuple(unravel(x, self.bit_sizes) + unravel(y, self.bit_sizes)) for x, y in new_inputs]
        outputs = self.outputs + new_outputs
        pairs = zip(np.array(inputs, dtype=float), outputs)
        random.shuffle(pairs)
        err = 0
        for (i, target) in pairs:
            output = network.feed(i)
            output = pack(output[-self.bit_sizes*2:])
            err += abs(target - output)/target
        score = 1 - (err/len(pairs))
        return {'fitness': score}

    def solve(self, network):
        """ Defines if a network has solved the problem. Size matters more than accuracy, but error should not exceed a certain margin. """
        return int(self.evaluate(network) >= self.min_accuracy)

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
            return NEATGenotype(inputs=p.bitsize*2, outputs=p.bitsize*2, weight_range=p.weight_range, types=[p.neuron_type])
        with open(p.runs_file, "a") as r:
            r.write('type,x,y,product')
            pop = NEATPopulation(genotype, popsize=p.popsize)
            task = MultiplicationTask(p.run_length, p.bitsize, p.accuracy_threshold, r)
            pop.epoch(generations=p.generations, evaluator=task, solution=task) #Run the task code
            with open(p.ann_file, "wb") as a:
                cPickle.dump(pop, a, -1) #Serialize the population to a file
