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

survival_features_svm = ['SibSp', 'Parch', 'Fare', 'female', 'male','Pclass_1', 'Pclass_2', 'Pclass_3',
                        'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                        'small_family', 'solo', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_SVM',
                        'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F',
                        'Deck_FE', 'Deck_FG', 'Deck_G', 'Unknown']

# ==================== Support Vector Machine (SVM) ==================== #

svclassifier = SVC(kernel="linear",
                   random_state=1234)

# //--  With median age  \\-- #

survived_median = cross_val_score(estimator=svclassifier,
                                  X=train.loc[:, survival_features_svm],
                                  y=train.loc[:, 'Survived'],
                                  cv=10,
                                  n_jobs=2)

print("The MEAN CV score is", round(survived_median.mean(), ndigits=2))
print("The standard deviation is", round(survived_median.std(), ndigits=2))
# The MEAN CV score is 0.82
# The standard deviation is 0.02

## Fit the model

svclassifier.fit(train.loc[:, survival_features_svm], train.loc[:, 'Survived'])
test.loc[:, "Survived"] = svclassifier.predict(test.loc[:, survival_features_svm]).astype(int)
SVM_test_basic = test.loc[:, ["PassengerId", "Survived"]]
SVM_test_basic.to_csv("titanic_submissions/survival_prediction_svm.csv", index=False)