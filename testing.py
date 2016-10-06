from sklearn import datasets as ds
import numpy as np
import matplotlib.pyplot as plt

iris = ds.load_iris()
plt.hist(iris.data, bins='scott')
plt.title("test")
plt.show()