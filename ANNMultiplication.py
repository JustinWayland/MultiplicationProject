from deps.peas.networks.rnn import NeuralNetwork
from deps.peas.methods.neat import NEATPopulation, NEATGenotype
from itertools import repeat
import random
import cPickle

def unravel(n, min_length, zero=0):
    output = []
    while n > 0:
        output.append(float(n & 1))
        n = n >> 1
    if len(output) < min_length:
        output.extend(repeat(n, min_length - len(output)))
    return output

def pack(a):
    b = 1
    s = 0
    for bit in a:
        if bit > 0:
            s += b
        b += b
    return s

class MultiplicationTask(object):
    def __init__(self, run_length, bit_sizes, accuracy_threshold=0.99):
        #Create fixed runs
        inputs = [(random.getrandbits(bit_sizes), random.getrandbits(bit_sizes)) for i in range(run_length)]
        self.inputs = [map(unravel, item, repeat(bit_sizes)) for item in inputs]
        self.outputs = [a*b for a, b in inputs]
        self.bit_sizes = bit_sizes
        self.run_length = run_length
        self.min_accuracy = accuracy_threshold

    def evaluate(self, network):
        network.make_feedforward()
        if not network.node_types[-1](-1000) < -0.95:
            raise Exception("Network should be able to output value of -1, e.g. using a tanh node.")
        #Create random runs (so as to prevent overfitting)
        new_inputs = [(random.getrandbits(self.bit_sizes), random.getrandbits(self.bit_sizes)) for i in range(self.run_length)]
        new_outputs = [a*b for a, b in new_inputs]
        pairs = zip(self.inputs + [map(unravel, t) for t in new_inputs], self.outputs + map(unravel, new_outputs))
        random.shuffle(pairs)
        err = 0
        for (i, target) in pairs:
            output = network.feed(i)
            output = pack(output[-bit_sizes:])
            err += abs(target - output)/target
        score = 1 - (err/len(pairs))
        return score

    def solve(self, network):
        """ Defines if a network has solved the problem. Size matters more than accuracy, but error should not exceed a certain margin. """
        return (self.evaluate(network) >= self.min_accuracy)

bitsize = 512 #Adjustable
    
def genotype():
    return NEATGenotype(inputs=bitsize*2, outputs=bitsize*2, types=['ident'])

popsize = 250 #Adjustable

pop = NEATPopulation(genotype, popsize=popsize)

run_length = 30 #Adjustable

task = MultiplicationTask(run_length, bitsize)

generations = 100 #Adjustable

pop.epoch(generations=generations, evaluator=task, solution=task) #Run the task code

filename = "testrun.ann" #Adjustable

with open(filename, "wb") as f:
    cpickle.dump(pop, f, -1) #Serialize the population to a file
