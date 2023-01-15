# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 00:42:41 2022

@author: aayus
"""

#!/usr/bin/env python3


from ML_Assignment import data_classification
import sys



# path here is the directory where the python file with function data classification is saved


# clf_opt chooses from {"logistic regression":"lr",
#                       "KNearest Neighbor":"knn",
#                       "Support Vector Machine":"svm",
#                       Multinomial Naive Bayes":"nb"}

print('{"\nlogistic regression":"lr", "KNearest Neighbor":"knn","Support Vector Machine":"svm", "Multinomial Naive Bayes":"nb"}')
p = input('\n  *****************Enter the classifier to use************************ : ')
if p == 'lr':
    clf=data_classification(path = "D:/VS Code/", clf_opt='lr') 
elif p == 'knn':
    clf=data_classification(path = "D:/VS Code/", clf_opt='knn')
elif p == 'svm':
    clf=data_classification(path = "D:/VS Code/", clf_opt='svm')
elif p == 'nb':
    clf=data_classification(path = "D:/VS Code/", clf_opt='nb')
else:
    print("################## Run the program again with a valid classifier #####################")
    sys.exit(0)
#Put the files and code in same directory to run the code

clf.classification()

    