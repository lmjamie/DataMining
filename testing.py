from sklearn import datasets as ds
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d as bs2

iris = ds.load_iris()
d =[]
ed = []
for items in range(iris.data.shape[1]):
    hist, edges = np.histogram(iris.data[:, items], bins=3)
    # this just stores the values we used to bin things
    ed.append(edges.tolist())
    d.append(np.digitize(iris.data[:, items], edges).tolist())
print(ed)
de = list(map(lambda i: (list(map(lambda x: x[i], d))), range(iris.data.shape[0])))
print(np.asarray(de))
