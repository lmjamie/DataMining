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


xt = []
def run_networks(set_neurons, sample):
    co_values = list(map(lambda x, y: x * y, get_weight(set_neurons), sample))
    if sum(co_values) < 0:
        for idx, items in enumerate(set_neurons):
            items.weight += .25*(2 - 1)
    xt.append(sum(get_weight(set_neurons)))


iris = ds.load_iris()
xy = make_nodes(4)
print(get_weight(xy))
for i in range(0, 1000):
    run_networks(xy, iris.data[0])
print(sum(get_weight(xy)))
plt.plot(xt)
plt.show()