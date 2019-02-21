# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''Attribute Information: (class attribute has been moved to last column)

   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
  '''

# Import packages  
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import os

os.system('clear')


# Load dataset
cancer_data = pd.read_csv("../data/breast-cancer-wisconsin.data", header=None)

#print(cancer_data.count())

cancer_data = cancer_data[cancer_data[6] != '?']
#print(cancer_data.count())

# Prepare data
X = cancer_data.iloc[:,1:10]
y = cancer_data.iloc[:,10]


# Divide data into training/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#print(X_train.count())
#print(X_test.count())

# Train model
lm = LinearRegression()
lm.fit(X_train, y_train)

#print('Coefficients: \n', lm.coef_)


# Test the Model
y_pred = [int(2) if item < 3 else int(4) for item in lm.predict(X_test)]
y_test = y_test.tolist()

print('MSE:', metrics.mean_squared_error(y_test, y_pred))


# Compute confussion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

plt.figure()
plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Breast Cancer Classification')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['P', 'N'])
plt.yticks(tick_marks, ['P', 'N'])

for i, j in it.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])) :
    plt.text(j,i,cnf_matrix[i][j], horizontalalignment="center", color="white" if cnf_matrix[i,j] > cnf_matrix.max()/2 else "black")

plt.ylabel('True Label')
plt.xlabel('Prdicted Label')
#plt.tight_layout()
plt.grid('off')
plt.show()

