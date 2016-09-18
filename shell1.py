from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
import sys
iris = datasets.load_iris()

class Classifier:
    def __init__(self):
        self.data = iris.data
        self.target = iris.target

    def train(self, train_data, train_target):
        print("And thus, the Machine was trained.")

    def predict_single(self, test_instance):
        return 0

    def predict(self, test_data):
        results = []
        for i in test_data:
            results.append(self.predict_single(test_data[i]))
        return results

def get_split_size(success=False):
    split_size = .7
    while not success:
        try:
            split_size = float(input("Please enter a decimal value for the train set size. (e.g. .7 or .85)\n>> "))
            success = True if split_size > 0 and split_size < 1 else False
            if not success:
                print("Error: Value entered was not between 1 and 0")
        except ValueError:
            print("Error: Value entered was not a decimal value")
    return tts(iris.data, iris.target, train_size=split_size, random_state=20)


def get_accuracy(results, test_targets):
    num_correct = 0
    for i in range(test_targets.size):
        num_correct += results[i] == test_targets[i]
    print("Predicted ", num_correct, " of ", test_targets.size,
          "\nFor an accuracy of {0:.2f}%".format(100 * (num_correct / test_targets.size)), sep="")

def main(argv):
    option = int(input(
        "Please select one of the options:\n1 - Choose your own size for the train set\n2 - Default size 70%\n>> "))
    iris_training, iris_test, itraining_target, itest_target = get_split_size() if option == 1 else get_split_size(success=True)
    hardcoded = Classifier()
    hardcoded.train(iris_training, itraining_target)
    get_accuracy(hardcoded.predict(itest_target), itest_target)

if __name__ == '__main__':
    main(sys.argv)

