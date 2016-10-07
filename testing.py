from sklearn import datasets as ds
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d as bs2

iris = ds.load_iris()
plt.hist(iris.data[:, 0], bins='scott')
plt.title("test")
plt.show()
