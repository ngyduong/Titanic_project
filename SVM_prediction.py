# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #

import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# ==================== DATA MANIPULATION ==================== #

test = pd.read_csv("titanic_data/clean_data/Clean_test.csv")
train = pd.read_csv("titanic_data/clean_data/Clean_train.csv")

# //--  Create the independent variables  \\-- #

survival_features_rf = ['SibSp', 'Parch', 'Fare', 'female', 'male','Pclass_1', 'Pclass_2', 'Pclass_3',
                        'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                        'small_family', 'solo', 'C', 'Q', 'S', 'Age_Randomforest']

# ==================== Support Vector Machine (SVM) ==================== #

svclassifier = SVC(kernel="linear",
                   shrinking=True,
                   probability=True)

# //--  With median age  \\-- #

survived_median = cross_val_score(estimator=svclassifier,
                                  X=train.loc[:, survival_features_rf],
                                  y=train.loc[:, 'Survived'],
                                  cv=10,
                                  n_jobs=2)

print("The MEAN CV score is", round(survived_median.mean(), ndigits=2))
print("The standard deviation is", round(survived_median.std(), ndigits=2))
# The MEAN CV score is 0.82
# The standard deviation is 0.03

## Fit the model

svclassifier.fit(train.loc[:, survival_features_rf], train.loc[:, 'Survived'])
test.loc[:, "Survived"] = svclassifier.predict(test.loc[:, survival_features_rf]).astype(int)
SVM_test_basic = test.loc[:, ["PassengerId", "Survived"]]
SVM_test_basic.to_csv("titanic_submissions/SVM_test.csv", index=False)