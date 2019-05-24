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
#from sklearn import svm
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
fileDates =data['date']           #store dates
data= data.drop('date',axis=1)    # remove date column
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

data.plot(kind='density', subplots=True, sharex=False,layout=(3,4),)
pyplot.show()
data.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
pyplot.show()
#'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm


classGamma =.001
classNu = 0.1

frontier_offset =3


# fit the model
print("-> Validator starting  (using Non-linear SVM Novelty classfier)")
clf = svm.OneClassSVM(nu=classNu, kernel="rbf", gamma=classGamma)
clf.fit(data)
y_pred_train = clf.predict(data)

prediction= list(zip(fileDates,y_pred_train ))
for x in prediction:
    print (x)



y_pred_test = clf.predict(data)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
print ("Machine has been trained..  \n\n\n\ Ready to evaluate....")
###############################################################################test the data
testInput = "testfile.csv"
testData = read_csv(testInput, header=0)
testFileDates =testData['date']           #store dates
testData= testData.drop('date',axis=1)    # remove date column
testfileCols  = testData.columns          # grab col names for re-insersion after normalization
print ("columns are: ",testfileCols)
print ("raw  data....")
print(testData)
scaler = MinMaxScaler(feature_range=(0, 1))   #change scale to 0-1 range to make the machine happy :-)
testData = scaler.fit_transform(testData)
print ("0-1 normalized  data....")
testscaler = StandardScaler().fit(testData)     #0 mean 1, std deviation
testData = testscaler.transform(testData)
print (testData)
#'''''
testData = pd.DataFrame(data=testData, columns=fileCols)    # change normalized data back into pandas data frame, reinsert columns
set_option('display.width', 550)
set_option('precision', 3)

Testdescription = testData.describe()
print ( "\n\file description =\n" ,  Testdescription)

print ( "\n\fileshape =" ,  testData.shape)
print('--')
print ("normalized and framed EVALUATION data....")
print(testData)
print('--')
#scatter_matrix(data)

testData.plot(kind='density', subplots=True, sharex=False,layout=(3,4),)
pyplot.show()

#'''
print(__doc__)