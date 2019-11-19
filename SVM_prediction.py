# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #


import pandas as pd
import numpy as np
import sklearn

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==================== IMPORT MANIPULATION ==================== #


test = pd.read_csv("titanic_data/Clean_test.csv")
train = pd.read_csv("titanic_data/Clean_train.csv")

# //--  Create the independent variables  \\-- #

survival_features = ['Age', 'SibSp', 'Parch', 'Fare', 'female', 'male',
                     'Pclass_1', 'Pclass_2', 'Pclass_3', 'Dr', 'Master',
                     'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                     'small_family', 'solo', 'C', 'Q', 'S']

# //--  Split the train dataset into train_x and train_y  \\-- #

train_X = train.loc[:, survival_features]
train_Y = train.loc[:, "Survived"]

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234)

# ==================== Support Vector Machine (SVM) ==================== #


# //--  Simple SVM linear model  \\-- #

svclassifier = SVC(kernel="linear")

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print("The accuracy score is", round(accuracy_score(y_test, y_pred), ndigits=2))

# The accuracy score is 0.85

svclassifier.fit(train.loc[:, survival_features], train.loc[:, 'Survived'])
test.loc[:, "Survived"] = svclassifier.predict(test.loc[:, survival_features]).astype(int)
SVM_test_basic = test.loc[:, ["PassengerId", "Survived"]]
SVM_test_basic.to_csv("titanic_submissions/SVM_test_basic.csv", index=False)