from sklearn import datasets as ds
from random import triangular
from pandas import read_csv as pd


class Neurons:
    def __init__(self):
<<<<<<< HEAD
        self.weight = None
        self.treshold = 0
=======
        self.weight = random.triangular(-2.0, 3.0)
        self.threshold = 0
>>>>>>> 66f2b67e7567f873e81797e1ce7e0142aad6ebd7
        self.is_active = False

    def bias(self):
        self.weight = -1

    def set_weight(self, new_size):
        self.weight = [triangular(-2.0, 3.0) for i in range(new_size)]


def make_nodes(size):
    return [Neurons() for _ in range(size)]


def get_weight(set_neurons):
    return [i.weight for i in set_neurons]


def preceptron(nodes_x, sample):
    neuron = make_nodes(nodes_x)
    for items in neuron:
        items.set_weight(sample.shape[0])
        items.weight.append(-1.00)
    co_values = []
    for items in neuron:
        co_values.append(list(map(lambda x, y: x * y, items.weight, sample)))
    for items in co_values:
        if sum(items) > 0:
            print("Result: 1")
        else:
            print("Result: 0")


def load_data():
    user_choice = int(input("1.Iris Data Set\n2.Pima Indian Diabetes Data Set\n>> "))

    if user_choice == 1:
        iris = ds.load_iris()
        target = iris.target
        data = iris.data
        for i in range(0, 49):
            preceptron(5, iris.data[i])

    if user_choice == 2:
        diabetes = pd("./diabetes.csv",
                               names=["Pregnant", "Plasma", "BloodPressure", "Triceps", "Insulin", "Mass", "Pedigree",
                                      "Age", "target"], dtype=float)
        data = diabetes.ix[:, :-1].values
        target = diabetes.target
        print(data)
        for i in data:
            preceptron(5, i)


load_data()
