# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


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
train_Y = train.loc[:, ["Survived", "PassengerId"]]

# ==================== Support Vector Machine (SVM) ==================== #

