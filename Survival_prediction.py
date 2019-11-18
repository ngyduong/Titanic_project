# ==================== PACKAGES ==================== #


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn import metrics


# ==================== IMPORT MANIPULATION ==================== #


test = pd.read_csv("titanic_data/Clean_test.csv")
train = pd.read_csv("titanic_data/Clean_train.csv")

# //--  Split the train dataset into train_x and train_y  \\-- #
train_X = train.loc[:, train.columns != "Survived"]
train_Y = train.loc[:, ["Survived", "PassengerId"]]

# //--  Create the independent variables  \\-- #

Survived_IV = [
               'Pclass', 'SibSp', 'Parch',
               'Fare',
               'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer',
               'big_family', 'small_family', 'solo',
               'female', 'male',
               'C', 'Q', 'S',
               '0_16', '17_24', '25_30', '31_40', 'over_40',
               'Age'
               ]

# ==================== RANDOM FOREST ==================== #

rfModel_Survived = RandomForestClassifier()

survived_accuracies = cross_val_score(estimator=rfModel_Survived,
                                      X=train_X.loc[:, Survived_IV],
                                      y=train_Y.loc[:, 'Survived'],
                                      cv=10,
                                      n_jobs=2)


print("The MEAN CV accuracy of survived prediction is", round(survived_accuracies.mean(), ndigits=2))
print("The MAX CV accuracy of survived prediction is", round(survived_accuracies.max(), ndigits=2))
print("The MIN CV accuracy of survived prediction is", round(survived_accuracies.min(), ndigits=2))