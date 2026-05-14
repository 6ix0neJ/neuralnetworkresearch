import math
import random

# random weights
weights = [random.random(), random.random()]
bias = random.random()

# sigmoid activation
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# neuron
def neuron(inputs):
    total = 0

    for i in range(len(inputs)):
        total += inputs[i] * weights[i]

    total += bias

    return sigmoid(total)

print(neuron([0.5, 0.8]))
