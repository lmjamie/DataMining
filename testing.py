from sklearn import datasets as ds
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d as bs2

iris = ds.load_iris()
d =[]
for items in range(iris.data.shape[1]):
    hist, edges = np.histogram(iris.data[:, items], bins=3)
    d.append(np.digitize(iris.data[:, items], edges))

