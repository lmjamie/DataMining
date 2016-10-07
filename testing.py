from sklearn import datasets as ds
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d as bs2

iris = ds.load_iris()
<<<<<<< HEAD
cancer = ds.load_breast_cancer()
print(list(map(list, cancer.data)))
# plt.hist(iris.data, bins=ValueError: too many values to unpack (expected 2)'scott')
# plt.title("test")
#
# plt.show()
=======
plt.hist(iris.data[:, 0], bins='scott')
plt.title("test")
plt.show()
>>>>>>> 427ac96699898b9641ae3529a31182fef479aa94
