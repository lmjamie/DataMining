from sklearn import datasets as ds
from random import triangular
from pandas import read_csv as pd
from numpy import array as array
from scipy.special import expit


class Neurons:
    def __init__(self, size, is_bias):
        self.weight, self.weight, self.threshold, self.is_active = None, [triangular(-2.0, 3.0) for _ in
                                                                          range(size)] if not is_bias else [
            -1.00], 0, False

    def sigmoid(self, item_list):
        return expit(sum(list(map(lambda x, y: x * y, self.weight, item_list))))


iris = ds.load_iris()
data = iris.data


def make_neural_row(attribute_size, size):
    neuro = [Neurons(attribute_size, False) for _ in range(size)]
    neuro.append(Neurons(attribute_size, True))
    return neuro


def make_net(size):
    net = []
    for i in range(size):
        neural_size = int(input("What size is this layer?\n>>"))
        net.append(make_neural_row(iris.data.shape[1] if i == 0 else len(net[i - 1]), neural_size))
    return net


n = make_net(2)
j = []
for idx, items in enumerate(n):
    j.append([k.sigmoid(iris.data[0] if idx == 0 else j[idx - 1]) for k in items])
print(j[-1])

guess = j[-1].index(max(j[-1]))

print(guess)