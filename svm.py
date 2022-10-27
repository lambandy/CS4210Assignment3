#-------------------------------------------------------------------------
# AUTHOR: Andy Lam
# FILENAME: CS4210Assignment3
# SPECIFICATION: You will simulate a grid search, trying to find which combination of four SVM hyperparameters (c, degree, kernel, and decision_function_shape) leads you to the best prediction performance.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
highestAccuracy = 0.0
for x in c: #iterates over c
    for y in degree: #iterates over degree
        for z in kernel: #iterates kernel
           for i in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here
                clf = svm.SVC(C=x, degree=y, kernel=z, decision_function_shape=i)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here
                accuracyCount = 0
                currentAccuracy = 0.0
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    classPredicted = clf.predict([x_testSample])
                    if int(classPredicted) == y_testSample:
                        accuracyCount += 1
                        currentAccuracy = accuracyCount/len(X_test)
                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if currentAccuracy > highestAccuracy:
                    highestAccuracy = currentAccuracy
                    print("Highest SVM accuracy so far: %f, Parameters: a=%d, degree=%d, kernel= %s, decision_function_shape = %s" %(highestAccuracy, x, y, z, i))



