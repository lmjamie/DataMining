from sklearn import datasets as ds
from random import randint as rand
from random import triangular as tri
from pandas import read_csv as r_csv
import numpy as np
from collections import Counter as co
from sklearn.cross_validation import train_test_split as tts
from scipy.special import expit
from sys import argv


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [tri(-1.0, 1.0) for _ in range(num_inputs + 1)]
        self.threshold = 0
        self.bias = -1

    def output(self, inputs):
        inputs = np.append(inputs, [self.bias])
        return self.sigmoid(inputs)

    def sigmoid(self, inputs):
        return expit(sum([self.weights[i] * x for i, x in enumerate(inputs)]))

class Node:
    def __init__(self, feature_name, child_nodes):
        self.feature_name = feature_name
        self.child_nodes = child_nodes


def height(node):
    return 0 if not isinstance(node, Node) else \
        np.max(list(map(lambda x: height(node.child_nodes[x]) + 1, node.child_nodes.keys())))


def print_level_order(root):
    for i in range(1, height(root) + 1):
        print("Level", i)
        print_given_level(root, i)


def print_given_level(root, level):
    if level == 1:
        print("Node", root.feature_name, "with branches", list(root.child_nodes.keys()))
    else:
        for next_node in root.child_nodes.values():
            print_given_level(next_node, level - 1)


class HardCodedClassifier:
    def __init__(self):
        self.classes = self.target = self.data = None

    def train(self, train_data, train_target):
        print("And thus, the Machine was trained.")

    def predict_single(self, test_instance):
        return 0

    def set_classes(self, classes):
        self.classes = classes

    def predict(self, test_data):
        return [self.predict_single(inst) for inst in test_data]


class NeuralNetworkClassifier(HardCodedClassifier):
    def __init__(self):
        super(HardCodedClassifier).__init__()
        self.network_layers = self.mean = self.std = self.num_attr = self.l_rate = None

    def train(self, train_data, train_target):
        self.mean, self.std, self.l_rate = train_data.mean(), train_data.std(), 0.2
        self.data, self.target = self.standardize(train_data), train_target
        self.num_attr = self.data.shape[1]
        self.make_network(int(input("How many hidden layers would you like?\n>> ")))
        results = self.get_results(self.data[1])
        self.update(1, self.data[1], results)
        print(results[-1], self.target[1])
        results = self.get_results(self.data[1])
        print(results[-1], "Now")
        input("holding")
        # for epoch in range(int(input("How many Epochs would you like?\n>>"))):
        #     pass

    def get_num_nodes(self, layer, num_layers):
        return int(input("How many Neurons would you like in hidden layer " + str(
            layer + 1) + "?\n>> ") if layer < num_layers else len(self.classes))

    def num_inputs(self, layer):
        return len(self.network_layers[layer - 1]) if layer > 0 else self.num_attr

    def make_network(self, num_layers):
        self.network_layers = []
        for i in range(num_layers + 1):
            self.network_layers.append(self.make_layer(self.num_inputs(i), self.get_num_nodes(i, num_layers)))

    def make_layer(self, num_inputs, num_nodes):
        return [Neuron(num_inputs) for _ in range(num_nodes)]

    def get_results(self, inputs):
        results = []
        for index, layer in enumerate(self.network_layers):
            results.append([n.output(results[index - 1] if index > 0 else inputs) for n in layer])
        return results

    def update(self, row, f_inputs, results):
        errors = [self.get_output_error(r, 0 if self.target[row] != np.argmax(results[-1]) else 1) for r in results[-1]]
        for layer_num, layer in reversed(list(enumerate(self.network_layers))):
            for i in range(len(self.network_layers[layer_num])):
                self.network_layers[layer_num][i].weights = self.update_weights(
                    self.network_layers[layer_num][i], results[layer_num - 1] if layer_num > 0 else f_inputs, errors)
            errors = [self.get_hidden_error(r, n.weights, errors) for r in results[layer_num] for n in layer]

    def update_weights(self, neuron, inputs, errors):
        return [w - self.l_rate * i * e for w in neuron.weights for i in inputs for e in errors]

    def get_output_error(self, result, target):
        return result * (1 - result) * (result - target)

    def get_hidden_error(self, result, f_weights, errors):
        return result * (1 - result) * sum([fw * e for fw in f_weights for e in errors])

    def predict_single(self, test_instance):
        results = self.get_results(self.standardize(test_instance))
        return np.argmax(results[-1])

    def standardize(self, data):
        return (np.asarray(data) - self.mean) / self.std


class DecisionTreeClassifier(HardCodedClassifier):
    def __init__(self):
        super(HardCodedClassifier).__init__()
        self.tree = None

    def train(self, train_data, train_target):
        self.data = train_data
        self.target = train_target
        self.tree = self.make_tree(range(train_data.shape[1]), range(train_target.shape[0]))

    def predict_single(self, test_instance):
        node = self.tree
        while isinstance(node, Node):
            node = node.child_nodes[test_instance[node.feature_name]]
        return node

    def make_tree(self, features_left, indices):
        # get list of each class result for the indices
        classes_list = list(map(lambda ind: self.target[ind], indices))

        # base case if there is only one unique class in list
        if np.unique(classes_list).size == 1:
            return classes_list[0]

        # base case if there are no more features
        if len(features_left) == 0:
            return co(classes_list).most_common(1)[0][0]

        # which feature has the lowest entropy
        best_feature = features_left[self.best_info_gain(features_left, indices)]

        # Get the unique possible values for this best feature
        values_of_feature = np.unique(self.data[:, best_feature])

        # For each possible value get the list of indices that have those values of the indices we are checking
        value_indices = list(map(
            lambda val: [ind for ind in indices if self.data[:, best_feature][ind] == val], values_of_feature))

        # if any of those possible values had no indices associated with it, then make it return the most common class
        for i in range(len(value_indices)):
            if len(value_indices[i]) == 0:
                value_indices[i] = [self.target.tolist().index(co(classes_list).most_common(1)[0][0])]

        # remove the best feature from the list of features left
        remaining = [i for i in features_left if i != best_feature]

        # make a node with a dictionary as children.
        return Node(best_feature,
                    {x: self.make_tree(remaining, y) for x in values_of_feature for y in value_indices})

    def best_info_gain(self, features, indices):
        """
        This function will calculate figure out which feature has the most info gain
        :param features: The features you have to look between
        :param indices: Which spots we are looking at right now
        :return: The index of the feature with the best info gain or least entropy
        """
        return np.argmin(list(map(lambda feature: self.entropy_of_feature(feature, indices), features)))

    def entropy_of_feature(self, feature, indices):
        # get the possible values for the feature
        values_of_feature = np.unique(self.data[:, feature])

        # get the indices of each of those values
        value_indices = list(map(
            lambda val: [ind for ind in indices if self.data[:, feature][ind] == val], values_of_feature))
        cnt = co()
        # total number for weighted average
        num_total = sum(map(lambda val_i: len(val_i), value_indices))
        total_entropy = 0

        for vi in value_indices:
            num = len(vi)
            for i in vi:
                cnt[self.target[i]] += 1
            total_entropy += (num / num_total) * sum(map(
                lambda c: -cnt[c] / num * np.log2(cnt[c] / num) if cnt[c] != 0 else 0, self.classes))
            cnt.clear()
        return total_entropy


class KGNClassifier(HardCodedClassifier):
    """K-GoodNeighbors, a nearest neighbors algorithm"""
    def __init__(self):
        super(HardCodedClassifier).__init__()
        self.k = self.mean = self.std = None

    def knn(self, inputs):
        distances = np.sum((self.data - inputs)**2, axis=1)
        indices = np.argsort(distances, axis=0)
        neighbor_classes = np.unique(self.target[indices[:self.k]])
        if len(neighbor_classes) == 1:
            closest = np.unique(neighbor_classes)[0]
        else:
            counts = np.zeros(max(neighbor_classes) + 1)
            for i in range(self.k):
                counts[self.target[indices[i]]] += 1
            closest = np.argmax(counts)
        return closest

    def predict_single(self, test_instance):
        s_test = self.standardize(test_instance)
        return self.knn(s_test)

    def train(self, train_data, train_target):
        k = 3
        if input("Choose your own k? (y/n):\n>> ") is "y" or "Y":
            k = int(input("Enter an odd k value:\n>> "))
        self.k = k if len(self.classes) % k != 0 else k + 2
        self.target = train_target
        x = np.asarray(train_data)
        self.mean = x.mean()
        self.std = x.std()
        self.data = self.standardize(train_data)

    def standardize(self, data):
        return (np.asarray(data) - self.mean) / self.std


def get_split_size(data_set, target_set, success=False):
    split_size = .7
    while not success:
        try:
            split_size = float(input("Please enter a decimal value for the train set size. (e.g. .7 or .85)\n>> "))
            success = True if 0 < split_size < 1 else False
            if not success:
                print("Error: Value entered was not between 1 and 0")
        except ValueError:
            print("Error: Value entered was not a decimal value")
    return tts(data_set, target_set, train_size=split_size, random_state=rand(1, 100000))


def process_data():
    classifier = get_classifier()
    which = int(input("Which dataset would you like?\n1 - Iris\n2 - Cars\n3 - Diabetes\n>> "))
    if isinstance(classifier, DecisionTreeClassifier) and which == 1:
        data, target, classes, de = get_dataset(which, isinstance(classifier, DecisionTreeClassifier))
    else:
        data, target, classes = get_dataset(which, isinstance(classifier, DecisionTreeClassifier))
    classifier.set_classes(classes)
    if want_cv():
        print("\nMean Accuracy: {:.2f}%".format(cross_validation(classifier, data, target)) + "\nBuilding final")
        classifier.train(data, target)
        if isinstance(classifier, DecisionTreeClassifier) and input(
                "Finished building tree\nWould you like to print? (y/n)\n>> ") == 'y':
            print_level_order(classifier.tree)
    else:
        option = int(input(
            "Please select one of the options:\n1 - Choose your own size for the train set\n2 - Default size 70%\n>> "))
        training, test, training_target, test_target = get_split_size(data, target, False if option == 1 else True)
        classifier.train(training, training_target)
        print("Accuracy: {:.2f}%".format(get_accuracy(classifier.predict(test), test_target)))


def want_cv():
    return True if input("Would you like to cross validate this? (y/n)\n>> ") == 'y' else False


def cross_validation(classifier, data, targets):
    training, train, target, tar = tts(data, targets, random_state=rand(1, 100000))
    training = np.append(training, train, axis=0)
    target = np.append(target, tar, axis=0)
    num_folds = int(input("How many folds would you like for cross validation?\n>> "))
    subset_size = len(data) // num_folds
    results = []
    for i in range(num_folds):
        print("\nRound", i + 1)
        testing_this_round = training[i * subset_size:][:subset_size]
        test_target_this_round = target[i * subset_size:][:subset_size]
        training_this_round = np.append(training[:i * subset_size], training[(i + 1) * subset_size:], axis=0)
        train_target_this_round = np.append(target[:i * subset_size], target[(i + 1) * subset_size:], axis=0)
        classifier.train(training_this_round, train_target_this_round)
        results.append(get_accuracy(classifier.predict(testing_this_round), test_target_this_round))
        print("Accuracy: {:.2f}%".format(results[i]))
    return np.asarray(results).mean()


def get_accuracy(results, test_targets):
    num_correct = 0
    for i in range(test_targets.size):
        num_correct += results[i] == test_targets[i]
    return 100 * (num_correct / test_targets.size)


def get_classifier():
    classifiers = {'1': NeuralNetworkClassifier,
                   '2': DecisionTreeClassifier,
                   '3': KGNClassifier,
                   '4': HardCodedClassifier}
    return classifiers[input(
        "Please select a Classifier:\n1 - Neural Network\n2 - Decision Tree\n3 - K-Nearest Neighbors\n"
        + "4 - Hardcoded\n>> ")]()


def get_iris(need_nominal=False):
    iris = ds.load_iris()
    if need_nominal:
        d = []
        ed = []
        for items in range(iris.data.shape[1]):
            hist, edges = np.histogram(iris.data[:, items], bins=3)
            # this just stores the values we used to bin things
            ed.append(edges.tolist())
            d.append(np.digitize(iris.data[:, items], edges).tolist())
        return np.asarray(list(map(lambda i: (
            list(map(lambda x: x[i], d))), range(iris.data.shape[0])))), iris.target, iris.target_names, ed
    return iris.data, iris.target, iris.target_names


def get_cars(need_nominal=False):
    my_read_in = r_csv("car.csv", dtype=str,
                       names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"])
    car_data = my_read_in.ix[:, :-1]
    car_target = my_read_in.target
    if not need_nominal:
        car_target = car_target.replace("unacc", 0).replace("acc", 1).replace("good", 2).replace("vgood", 3)
        car_data.buying = car_data.buying.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
        car_data.maint = car_data.maint.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
        car_data.doors = car_data.doors.replace("2", 2).replace("3", 3).replace("4", 4).replace("5more", 5)
        car_data.persons = car_data.persons.replace("2", 2).replace("4", 4).replace("more", 6)
        car_data.lug_boot = car_data.lug_boot.replace("small", 1).replace("med", 2).replace("big", 3)
        car_data.safety = car_data.safety.replace("low", 1).replace("med", 2).replace("high", 3)

    return car_data.values, car_target.values, ["unacc", "acc", "good", "vgood"]


def get_diabetes(need_nominal=False):
    read_in = r_csv("diabetes.csv", names=["pregnant", "plasma_glucose", "blood_pressure", "triceps", "insulin", "mass",
                                           "pedigree", "age", "target"], dtype=float)
    diabetes_data = read_in.ix[:, :-1].values
    diabetes_target = read_in.target.values

    return diabetes_data, diabetes_target, ["negative", "Positive"]


def get_dataset(which, need_nominal=False):
    dataset = {1: get_iris,
               2: get_cars,
               3: get_diabetes}
    return dataset[which](need_nominal)


def main(argv):
    process_data()


if __name__ == '__main__':
    main(argv)
