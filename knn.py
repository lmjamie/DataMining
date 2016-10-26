from hardcoded import HardCodedClassifier as hcc
import numpy as np


class KGNClassifier(hcc):
    """K-GoodNeighbors, a nearest neighbors algorithm"""
    def __init__(self):
        super(hcc).__init__()
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
