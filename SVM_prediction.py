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

survival_features_svm = ['SibSp', 'Parch', 'Fare', 'female', 'male','Pclass_1', 'Pclass_2', 'Pclass_3',
                        'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                        'small_family', 'solo', 'C', 'Q', 'S', 'Age_SVM']

survival_features_replace = ['SibSp', 'Parch', 'Fare', 'female', 'male','Pclass_1', 'Pclass_2', 'Pclass_3',
                        'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                        'small_family', 'solo', 'C', 'Q', 'S', 'Age_replace']

# ==================== Support Vector Machine (SVM) ==================== #


# # //--  Simple SVM linear model  \\-- #
#
# svclassifier = SVC(kernel="linear")
#
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
#
# print("The accuracy score is", round(accuracy_score(y_test, y_pred), ndigits=2))
#
# # The accuracy score is 0.85
#
# svclassifier.fit(train.loc[:, survival_features], train.loc[:, 'Survived'])
# test.loc[:, "Survived"] = svclassifier.predict(test.loc[:, survival_features]).astype(int)
# SVM_test_basic = test.loc[:, ["PassengerId", "Survived"]]
# SVM_test_basic.to_csv("titanic_submissions/SVM_test_basic.csv", index=False)