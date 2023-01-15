# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 00:42:41 2022

@author: aayus
"""

#!/usr/bin/env python3

import sys
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline

from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter


class data_classification():
     def __init__(self,path='C:/Users/aayus',clf_opt='lr'):
        self.path = path
        self.clf_opt=clf_opt
         

# Selection of classifiers  
     def classification_pipeline(self):    
  

    # Logistic Regression 
        if self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
       
    # Gaussian Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Naive Bayes Classifier ### \n')
            clf =  GaussianNB(priors=None)  
            clf_parameters = {
                #'var_smoothing': np.logspace(0,-9, num=100)
            }     

    # KNN
        elif self.clf_opt=='knn':
            print('\n\t ### Training KNN Classifier ### \n')
            clf = KNeighborsClassifier(weights='uniform')
            clf_parameters = {
            'clf__n_neighbors':(5,13,14,15,9,10),
            'clf__p':(1,2,3,4),
            
            }          
    # Support Vector Machine  
        elif self.clf_opt=='svm': 
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced',probability=True)  
            clf_parameters = {
            'clf__C':(0.1,1,100),
            #'clf__kernel':('linear','rbf','polynomial'),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters  
    
# Load the data 
     def get_data(self):
         #Load Data again using pandas
         train_X = pd.read_csv(self.path+"training_data.csv", header=None)
         train_Y = pd.read_csv(self.path+"training_data_class_labels.csv",header=None)
         test_X = pd.read_csv(self.path+"test_data.csv",header=None)
         
         plt.scatter(train_X[0],train_X[1],c=train_Y[0],cmap=plt.cm.Accent)
         plt.show()
         
         fig = plt.figure(figsize=(10,10))
         ax = plt.axes(projection='3d')
         ax.scatter(train_X[train_Y[0] == 0][0],train_X[train_Y[0] == 0][1],train_Y[train_Y[0] == 0])
         ax.scatter(train_X[train_Y[0] == 1][0],train_X[train_Y[0] == 1][1],train_Y[train_Y[0] == 1], 'r')
         plt.show()
         

         
         X = train_X.to_numpy()
         Y = train_Y.to_numpy().ravel()
         
         tst_X = test_X.to_numpy()
         
         trn_data = X
         tst_data = tst_X
         trn_cat = Y
 
         return trn_data, tst_data, trn_cat

         
    
# Classification using the Gold Statndard after creating it from the raw text    
     def classification(self):  
   # Get the data
        trn_data, tst_data, trn_cat=self.get_data()
        trn_data=np.asarray(trn_data)
        tst_data=np.asarray(tst_data)

# Experiments using training data only during training phase (dividing it into training and validation set)
        skf = StratifiedKFold(n_splits=10)
        predicted_class_labels=[]; actual_class_labels=[]; 
        count=0; probs=[];
        for train_index, test_index in skf.split(trn_data,trn_cat):
            X_train=[]; y_train=[]; X_test=[]; y_test=[]
            for item in train_index:
                X_train.append(trn_data[item])
                y_train.append(trn_cat[item])
            for item in test_index:
                X_test.append(trn_data[item])
                y_test.append(trn_cat[item])
            count+=1                
            print('Training Phase '+str(count))
            clf,clf_parameters=self.classification_pipeline()
            pipeline = Pipeline([
                #       ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),                         # k=1000 is recommended 
                #        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
                        ('clf', clf),])
            grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_macro',cv=10)          
            grid.fit(X_train,y_train)     
            clf= grid.best_estimator_  
            #print('\n\n The best set of parameters of the pipiline are: ')
            #print(clf)     
            predicted=clf.predict(X_test)  
            predicted_probability = clf.predict_proba(X_test)
            for item in predicted_probability:
                probs.append(float(max(item)))
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)           
        confidence_score=statistics.mean(probs)-statistics.variance(probs)
        confidence_score=round(confidence_score, 3)
        #print("\n Predicted Probability for test data is: \t"+str(probs))
        print('\n\n The best set of parameters of the pipeline are: ')
        print(clf) 
        print ('\n The Probablity of Confidence of the Classifier: \t'+str(confidence_score)+'\n') 

    # Evaluation
        class_names=list(Counter(trn_cat).keys())
        class_names = [str(x) for x in class_names] 
        print('\n\n The classes are: ')
        print(class_names)      

        print('\n *************** Confusion Matrix ***************  \n')
        print (confusion_matrix(actual_class_labels, predicted_class_labels))        
        print('\n ***************  Scores on Training Data  *************** \n ')
        print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names))        
        
        # Experiments on Given Test Data during Test Phase
        if confidence_score>0.85:
            print('\n ***** Classification of Test Data ***** \n')   
            predicted_class_labels=[];
            clf,clf_parameters=self.classification_pipeline()
            pipeline = Pipeline([
                #        ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),                         # k=1000 is recommended 
                #        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
                        ('clf', clf),])
            grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_macro',cv=10)          
            grid.fit(trn_data,trn_cat)     
            clf= grid.best_estimator_ 
            predicted=clf.predict(tst_data)
            #print('\n ***************  Scores on Test Data  *************** \n ')
            #print(classification_report(tst_cat, predicted, target_names=class_names)) 
            print('\n ***************  Predicted Values of the test data  *************** \n')
            #print('\n Class of the test data: \t'+ str(predicted))
            print(pd.DataFrame(predicted,columns=['Predicted']))
            #pd.DataFrame(predicted,columns=['Predicted']).to_csv('Predicted_using_'+self.clf_opt+'.csv')    
            np.savetxt('Predicted_using_'+self.clf_opt+".csv", predicted, delimiter=",")