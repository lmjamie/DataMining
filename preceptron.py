from sklearn import datasets as ds
from sklearn.neural_network import MLPClassifier
import random
import matplotlib.pyplot as plt


class Neurons:
    def __init__(self):
        self.weight = random.triangular(-2.0, 3.0)
        self.treshHold = 0
        self.is_active = False

    def bias(self):
        self.weight = -1

    def set_weight(self, new_weight):
        self.weight = new_weight


def make_nodes(size):
    return [Neurons() for i in range(0, size)]


def get_weight(set_neurons):
    return [i.weight for i in set_neurons]


def preceptron(amount, sample):
    set_neurons = make_nodes(amount)
    co_values = list(map(lambda x, y: x * y, get_weight(set_neurons), sample))
    # TODO: make return instead of print
    if sum(co_values) > 0:
        print("Result: 1")
    else:
        print("Result: 2")






