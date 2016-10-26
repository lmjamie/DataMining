from sklearn.cross_validation import train_test_split as tts
from pandas import read_csv as r_csv
from sklearn import datasets as ds
from random import randint as rand
import numpy as np
import knn
import hardcoded as hc
import decision_tree as dt
import neural_network as nn


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
    if isinstance(classifier, dt.DecisionTreeClassifier) and which == 1:
        data, target, classes, de = get_dataset(which, isinstance(classifier, dt.DecisionTreeClassifier))
    else:
        data, target, classes = get_dataset(which, isinstance(classifier, dt.DecisionTreeClassifier))
    classifier.set_classes(classes)
    if want_cv():
        print("\nMean Accuracy: {:.2f}%".format(cross_validation(classifier, data, target)) + "\nBuilding final")
        classifier.train(data, target)
        if isinstance(classifier, dt.DecisionTreeClassifier) and input(
                "Finished building tree\nWould you like to print? (y/n)\n>> ") == 'y':
            dt.print_level_order(classifier.tree)
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
    classifiers = {'1': nn.NeuralNetworkClassifier,
                   '2': dt.DecisionTreeClassifier,
                   '3': knn.KGNClassifier,
                   '4': hc.HardCodedClassifier}
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
