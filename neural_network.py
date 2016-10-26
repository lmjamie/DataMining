from random import triangular as tri
from hardcoded import HardCodedClassifier as hcc
import numpy as np
from scipy.special import expit
#

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [tri(-1.0, 1.0) for _ in range(num_inputs + 1)]
        self.threshold = 0
        self.bias = -1
        self.error = None

    def output(self, inputs):
        inputs = np.append(inputs, [self.bias])
        return self.sigmoid(inputs)

    def sigmoid(self, inputs):
        return expit(sum([self.weights[i] * x for i, x in enumerate(inputs)]))


class NeuralNetworkClassifier(hcc):
    def __init__(self):
        super(hcc).__init__()
        self.network_layers = self.mean = self.std = self.num_attr = self.l_rate = None

    def train(self, train_data, train_target):
        self.mean, self.std, self.l_rate = train_data.mean(), train_data.std(), 0.1
        self.data, self.target = self.standardize(train_data), train_target
        self.num_attr = self.data.shape[1]
        self.make_network(int(input("How many hidden layers would you like?\n>> ")))
        results = self.get_results(self.data[0])
        self.update(0, self.data[0], results)
        # for i, x in enumerate(self.data):
        #     results = self.get_results(x)
        #     print("Results", results[-1], "Target", self.target[i])
        #     self.update(i, x, results)
        #     results = self.get_results(x)
        #     print("Updated", results[-1])
        #     # for epoch in range(int(input("How many Epochs would you like?\n>>"))):
        #     #     pass

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
        self.update_errors(row, results)
        for i, layer in enumerate(self.network_layers):
            for n in layer:
                self.update_weights(n, results[i - 1] if i > 0 else f_inputs)

    def update_errors(self, row, results):
        for i_layer, layer in reversed(list(enumerate(self.network_layers))):
            for i_neuron, neuron in enumerate(layer):
                neuron.error = self.get_hidden_error(
                    results[i_layer][i_neuron], [nn.weights[i_neuron] for nn in self.network_layers[i_layer + 1]],
                    [nn.error for nn in self.network_layers[i_layer + 1]]) if i_layer < len(
                    results) - 1 else self.get_output_error(results[i_layer][i_neuron], i_neuron == self.target[row])

    def update_weights(self, neuron, inputs):
        print("Old weights:", neuron.weights)
        neuron.weights = [w - self.l_rate * i * neuron.error for w in neuron.weights for i in inputs]
        print("New weights", neuron.weights, "\n------------")

    def get_output_error(self, result, target):
        return result * (1 - result) * (result - target)

    def get_hidden_error(self, result, f_weights, errors):
        return result * (1 - result) * sum([fw * e for fw in f_weights for e in errors])

    def predict_single(self, test_instance):
        results = self.get_results(self.standardize(test_instance))
        return np.argmax(results[-1])

    def standardize(self, data):
        return (np.asarray(data) - self.mean) / self.std
