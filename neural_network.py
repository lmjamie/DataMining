from random import triangular as tri
from hardcoded import HardCodedClassifier as hcc
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


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
        self.mean, self.std = train_data.mean(), train_data.std()
        self.data, self.target = self.standardize(train_data), train_target
        self.num_attr = self.data.shape[1]
        self.l_rate = float(input("What learning rate would you like? (e.g 0.1 - 0.4)\n>> "))
        self.make_network(int(input("How many hidden layers would you like?\n>> ")))
        self.epoch_learn(int(input("How many Epochs would you like?\n>>")))

    def epoch_learn(self, num_epochs):
        accuracy = []
        error = []
        display = input("Would you like to display the Accuracy and the Error for this training? (y/n)\n>> ") == 'y'

        for epoch in range(num_epochs):
            predictions = []
            ss_error = []
            for d, t in zip(self.data, self.target):
                results = self.get_results(d)
                predictions.append(np.argmax(results[-1]))
                ss_error.append(sum([((i == t) - r) ** 2 for i, r in enumerate(results[-1])]))
                self.update(t, d, results)
            accuracy.append(100 * sum([self.target[i] == p for i, p in enumerate(predictions)]) / self.target.size)
            error.append(sum(ss_error) / len(ss_error))
            if display:
                print("Accuracy for Epoch {}: {:.5f}%\nAverage Sum Squared Error: {:.7f}\n---------------------".format(
                    epoch + 1, accuracy[epoch], error[epoch]))
        self.plot(accuracy, "Accuracy", num_epochs)
        self.plot(error, "Error", num_epochs)

    def plot(self, data, data_name, num_epochs):
        if input("Do you want to plot the graph for {}? (y/n)\n>> ".format(data_name)) == 'y':
            plt.plot(range(1, num_epochs + 1), data)
            plt.title("Training {}".format(data_name))
            plt.xlabel("Epoch")
            plt.ylabel(data_name)
            print("Showing graph for {}. Please close the graph to continue.\n".format(data_name))
            plt.show()
        else:
            print("Skipping graph for {}".format(data_name))

    def get_num_nodes(self, layer, num_layers):
        return int(input("How many Neurons would you like in hidden layer {}?\n>> ".format(layer + 1))
                   if layer < num_layers else len(self.classes))

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

    def update(self, target, f_inputs, results):
        self.update_errors(target, results)
        self.update_all_weights(f_inputs, results)

    def update_errors(self, target, results):
        for i_layer, layer in reversed(list(enumerate(self.network_layers))):
            for i_neuron, neuron in enumerate(layer):
                neuron.error = self.get_error(i_neuron, i_layer, target, results)

    def get_error(self, i_neuron, i_layer, target, results):
        return self.get_hidden_error(
            results[i_layer][i_neuron], self.get_f_weights(i_neuron, i_layer), self.get_f_errors(i_layer)) \
            if i_layer < len(results) - 1 else self.get_output_error(results[i_layer][i_neuron], i_neuron == target)

    def get_f_weights(self, i_neuron, i_layer):
        return [nn.weights[i_neuron] for nn in self.network_layers[i_layer + 1]]

    def get_f_errors(self, i_layer):
        return [nn.error for nn in self.network_layers[i_layer + 1]]

    def update_all_weights(self, f_inputs, results):
        for i, layer in enumerate(self.network_layers):
            for n in layer:
                self.update_weights(n, results[i - 1] if i > 0 else f_inputs.tolist())

    def update_weights(self, neuron, inputs):
        inputs = inputs + [-1]
        neuron.weights = [w - self.l_rate * inputs[i] * neuron.error for i, w in enumerate(neuron.weights)]

    def get_output_error(self, result, target):
        return result * (1 - result) * (result - target)

    def get_hidden_error(self, result, f_weights, errors):
        return result * (1 - result) * sum([fw * errors[i] for i, fw in enumerate(f_weights)])

    def predict_single(self, test_instance):
        results = self.get_results(self.standardize(test_instance))
        return np.argmax(results[-1])

    def standardize(self, data):
        return (np.asarray(data) - self.mean) / self.std
