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
