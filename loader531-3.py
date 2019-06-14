# Univariate Density Plots
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib.font_manager

VERBOSE = False  # set to True for debugging info
print ("Environment scan...")

# Check the versions of libraries
import sys
print('-> Python: {}'.format(sys.version))
print('-> matplotlib: {}'.format(matplotlib.__version__))
print('-> pandas: {}'.format(pd.__version__))
print('-> sklearn: {}'.format(sklearn.__version__))

# extend line width when printing
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

filename = 'samplefile.csv'             # input file.   I contains many training rows and
np.set_printoptions(linewidth=200)
####
data = read_csv(filename, header=0)
Data= data.loc[data['type'] == 'train']
fileDates = data['date']                 #store dates
fileTypes = data['type']                 #store Types
data      = data.drop('date' ,axis=1)    # remove date column
data      = data.drop ('type', axis =1)  # remove type column
fileCols  = data.columns                 # grab col names for re-insersion after normalization

print ("columns from input file: ",fileCols)
if VERBOSE:
    print ("------------------------------raw  data----------------------------------------------------------")
    print(data)
scaler = MinMaxScaler(feature_range=(0, 1))   #change scale to 0-1 range to make the machine happy :-)
data = scaler.fit_transform(data)
if VERBOSE:
    print ("-------------------------------transformed to 0-1 normalized  data....------------------------------------------")
    print (data)
scaler = StandardScaler().fit(data)     # 0 mean 1, std deviation
data = scaler.transform(data)
if VERBOSE:
    print ("-------------------------------transformed to 0 mean 1, std deviation...------------------------------------------")
    print (data)

data = pd.DataFrame(data=data, columns=fileCols)    # change normalized data back into pandas data frame, reinsert columns
set_option('display.width', 550)
set_option('precision', 1)
print ( "\nfileshape =" , data.shape)
set_option('display.float_format', '{:.2f}'.format)
print ( "\nfile description =\n" , data.describe())
if VERBOSE:
    print('--')
    print ("-----------------------normalized and framed  data....------------------------------------------")
    print(data)
    print('--')
#-----------------------------------DISPLAY TRAINING DATA ---------------------------------------

data.plot(kind='density', subplots=True, sharex=False,layout=(3,4))
pyplot.show()
visualData= data.to_numpy()

#now re-insert date and type into the frame
data['date']=fileDates
data['type']=fileTypes

trainData = data.loc[data['type'] == 'train']

X_train, X_test = train_test_split(trainData,  test_size=.33, random_state=4)  #split data into train and test data  (66% train data)
X_trainClean=X_train.iloc[:, :-2]   # [:, -2:] = use the entier set and drop last 2 cols ( date and type)
X_testClean=X_train.iloc[:, :-2]    # [:, -2:] = use the entier set and drop last 2 cols ( date and type)

classGamma      =.2
classNu         = 0.1
frontier_offset =3

# fit the model
print("-> Machine Training starting  (using Non-linear SVM Novelty classfier) <--" )

clf = svm.OneClassSVM(nu=classNu, kernel="rbf", gamma=classGamma)
clf.fit(X_trainClean)

x_pred_train = clf.predict(X_trainClean)

count=0
for index, row in X_train.iterrows():
    if x_pred_train[count] == -1:
        print ( row[10], ": File details are different enough from the other files that it will not be used as part of the training set "  )
    else:
        print(row[10], ".....ok ")
    count=count+1

print ("Machine has been trained.  Evaluating procution file..... ")
result1 = clf.score_samples(X_testClean)
print (" Training File score->>>> ", np.average (np.exp(result1)))
x = np.linspace(0, 1, len ( result1))  # create a list of x valuses so that we can scatter the results uniformely across the graph
pyplot.scatter (x, np.exp(result1),label = 'Training Files Score')
#-------------------------------TEST PRODUCTION FILE-------------------

testData = data.loc[data['type'] == 'test']
testClean=testData.iloc[:, :-2]
pred_test = clf.predict(testClean)    #evaluate production files
# n_error_train = x_pred_train[x_pred_train == -1].size
# n_error_test = y_pred_test[y_pred_test == -1].size
# print ("n_error_train",n_error_train)
# print ("n_error_test",n_error_test)
result = clf.score_samples(testClean)
print (" production File score->>>>", np.exp(result))
count =0
for index, row in testData.iterrows():
    if pred_test[count] == -1:
        print ("* * * DO NOT SEND TO IMATCH BEFORE INVESTIGATING * * *: file for ",row[10], " has failed its integrity test. "  )
    else:
        print("OK TO SEND TO IMATCH.  File for :", row[10], " has passsed the integrity test  .")
    count= count+1
pyplot. axhline(y=np.exp(result), color='red',label = 'Todays File Score')
plt.legend()
pyplot.show()