from sklearn import datasets as ds
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split as tts
import sys
pd.options.mode.chained_assignment = None


class HardCodedClassifier:
    classes = None
    target = None
    data = None

    def train(self, train_data, train_target):
        print("And thus, the Machine was trained.")

    def predict_single(self, test_instance):
        return 0

    def set_classes(self, classes):
        self.classes = classes

    def predict(self, test_data):
        results = []
        for i in test_data:
            results.append(self.predict_single(i))
        return results


class DecisionTreeClassifier(HardCodedClassifier):
    def info_gain(self, test_data):
        columns = []
        for i in range(self.data.shape[1]):
            columns.append(test_data[:, i])

        entropy = []
        for col in columns:
            entropy.append(sum(list(map(lambda val: -val * np.log2(val) if val != 0 else 0, col))))

        feature = np.argmax(entropy)
        values = np.unique(self.data[:, feature])
        print(values)


class KGNClassifier(HardCodedClassifier):
    """K-GoodNeighbors, a nearest neighbors algorithm"""
    k = None
    mean = None
    std = None

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
    return tts(data_set, target_set, train_size=split_size, random_state=20123)


def process_data():
    data, target, classes = get_dataset()
    classifier = get_classifier()
    option = int(input(
        "Please select one of the options:\n1 - Choose your own size for the train set\n2 - Default size 70%\n>> "))
    training, test, training_target, test_target = get_split_size(data, target, False if option == 1 else True)
    classifier.set_classes(classes)
    classifier.train(training, training_target)
    get_accuracy(classifier.predict(test), test_target)


def get_accuracy(results, test_targets):
    num_correct = 0
    for i in range(test_targets.size):
        num_correct += results[i] == test_targets[i]
    print("Predicted ", num_correct, " of ", test_targets.size,
          "\nFor an accuracy of {0:.2f}%".format(100 * (num_correct / test_targets.size)), sep="")


def get_classifier():
    which = int(input("Please select a Classifier:\n1 - Hardcoded\n2 - K-Nearest Neighbors\n>> "))
    return KGNClassifier() if which == 2 else HardCodedClassifier()


def get_dataset():
    which = int(input("Please choose a Dataset:\n1 - Iris\n2 - Cars\n>> "))
    if which == 1:
        iris = ds.load_iris()
        return iris.data, iris.target, iris.target_names
    else:
        my_read_in = pd.read_csv("car.csv", dtype=str,
                                 names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"])
        car_data = my_read_in.ix[:, :-1]
        car_target = my_read_in.target.replace("unacc", 0).replace("acc", 1).replace("good", 2).replace("vgood", 3)
        car_data.buying = car_data.buying.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
        car_data.maint = car_data.maint.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
        car_data.doors = car_data.doors.replace("2", 2).replace("3", 3).replace("4", 4).replace("5more", 5)
        car_data.persons = car_data.persons.replace("2", 2).replace("4", 4).replace("more", 6)
        car_data.lug_boot = car_data.lug_boot.replace("small", 1).replace("med", 2).replace("big", 3)
        car_data.safety = car_data.safety.replace("low", 1).replace("med", 2).replace("high", 3)

        return car_data.values, car_target.values, ["unacc", "acc", "good", "vgood"]


def main(argv):
    # process_data()
    d, t, ta = get_dataset()

    myClassifier = DecisionTreeClassifier()
    train, test, t_target, test_target = get_split_size(d, t, True)
    myClassifier.data = train
    myClassifier.info_gain(test)

    myClassifier

if __name__ == '__main__':
    main(sys.argv)