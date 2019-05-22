# Univariate Density Plots
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
filename = 'samplefile.csv'

data = read_csv(filename, header=0)
set_option('display.width', 550)
set_option('precision', 3)
description = data.describe()
print ( "file shape =" , description)
scatter_matrix(data)
pyplot.show()
data.plot(kind='density', subplots=True, sharex=False,layout=(3,4),)
pyplot.show()
data.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
pyplot.show()

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
train_size =100
test_size =20
outliers_size = 10


classGamma =.1
classNu = 0.1

frontier_offset =3


# fit the model
print("-> Validator starting  (using Non-linear SVM Novelty classfier)")
clf = svm.OneClassSVM(nu=classNu, kernel="rbf", gamma=classGamma)
clf.fit(data)
y_pred_train = clf.predict(data)
y_pred_test = clf.predict(data)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
print ("the end")

