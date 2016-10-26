from hardcoded import HardCodedClassifier as hcc
from collections import Counter as co
import numpy as np


class DecisionTreeClassifier(hcc):
    def __init__(self):
        super(hcc).__init__()
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