# Univariate Density Plots
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm

print ("Environment scan...")

# Check the versions of libraries

# Python version
import sys

print('-> Python: {}'.format(sys.version))
print('--> matplotlib: {}'.format(matplotlib.__version__))
print('---> pandas: {}'.format(pd.__version__))
print('----> sklearn: {}'.format(sklearn.__version__))

# extend line width when printing
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

filename = 'samplefile.csv'
np.set_printoptions(linewidth=200)
####
data = read_csv(filename, header=0)
fileDates = data['date']           #store dates
fileTypes = data['type']          #store Types
data=       data.drop('date',axis=1)    # remove date column
data =      data.drop ('type', axis =1) # remove type column
fileCols  = data.columns          # grab col names for re-insersion after normalization

print ("columns are: ",fileCols)
print ("raw  data....")
print(data)
scaler = MinMaxScaler(feature_range=(0, 1))   #change scale to 0-1 range to make the machine happy :-)
data = scaler.fit_transform(data)
print ("0-1 normalized  data....")
scaler = StandardScaler().fit(data)     #0 mean 1, std deviation
data = scaler.transform(data)
print (data)
#'''''
data = pd.DataFrame(data=data, columns=fileCols)    # change normalized data back into pandas data frame, reinsert columns
set_option('display.width', 550)
set_option('precision', 3)

description = data.describe()
print ( "\n\file description =\n" ,  description)

print ( "\n\fileshape =" ,  data.shape)
print('--')
print ("normalized and framed  data....")
print(data)
print('--')
#scatter_matrix(data)

data[:27].plot(kind='density', subplots=True, sharex=False,layout=(3,4),)
#pyplot.show()
print (data[28:])
pyplot.show()
#data.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
#pyplot.show()
#'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
X_train, X_test = train_test_split(data,  test_size=.33, random_state=4)  #split data into train and test data  (66% train data)
classGamma =.1
classNu = 0.1

frontier_offset =3


# fit the model
print("-> Machine Training  starting  (using Non-linear SVM Novelty classfier)")
clf = svm.OneClassSVM(nu=classNu, kernel="rbf", gamma=classGamma)
clf.fit(X_train)

#print (x)[for x in clf.get_params()]
x_pred_train = clf.predict(X_train)
prediction= list(zip(fileDates,x_pred_train))
for x in prediction:
    if x[1] == -1:
        print ( x[0], " File details are different enough from the other files that  will not be used as part of the training set "  )
    else:
        print(x[0], " ok ")

print ("Machine has been trained..  \n\n\n\ ")

result = clf.score_samples(X_test)
print (" score->>>> ", result)


y_pred_test = clf.predict(data[28:])    #evaluate production files
n_error_train = x_pred_train[x_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
print ("n_error_train",n_error_train)
print ("n_error_test",n_error_test)


result = clf.score_samples(data[28:])
print (" production score->>>> ", result)

print (fileDates[28])
for x in y_pred_test:
    if x == -1:
        print ("DO NOT SEND TO IMATCH BEFORE INVESTIGATING : file for ",fileDates[28], " has failed its integrity test. "  )
    else:
        print("OK TO SEND TO IMATCH.  File for date:", fileDates[28], " has passsed the integrity test  .")
