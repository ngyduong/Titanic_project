# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #

import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
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

# ==================== RANDOM FOREST ==================== #

rfModel_Survived = RandomForestClassifier()

# //--  CVS with Age predicted by random forest  \\-- #

age_rf = cross_val_score(estimator=rfModel_Survived,
                         X=train.loc[:, survival_features_rf],
                         y=train.loc[:, 'Survived'],
                         cv=10,
                         n_jobs=2)

print("The MEAN CV score is", round(age_rf.mean(), ndigits=2))
print("The standard deviation is", round(age_rf.std(), ndigits=2))
# The MEAN CV score is 0.8
# The standard deviation is 0.05

# //--  CVS with Age predicted by SVM  \\-- #

age_svm = cross_val_score(estimator=rfModel_Survived,
                          X=train.loc[:, survival_features_svm],
                          y=train.loc[:, 'Survived'],
                          cv=10,
                          n_jobs=2)

print("The MEAN CV score is", round(age_svm.mean(), ndigits=2))
print("The standard deviation is", round(age_svm.std(), ndigits=2))
# The MEAN CV score is 0.8
# The standard deviation is 0.05

# //--  CVS with Age replaced by median depending on title  \\-- #

age_replace = cross_val_score(estimator=rfModel_Survived,
                              X=train.loc[:, survival_features_replace],
                              y=train.loc[:, 'Survived'],
                              cv=10,
                              n_jobs=2)

print("The MEAN CV score is", round(age_replace.mean(), ndigits=2))
print("The standard deviation is", round(age_replace.std(), ndigits=2))
# The MEAN CV score is 0.81
# The standard deviation is 0.03

# The one with the best accuracy and least standard deviation is the one in which we replace
# the Age with the median age depending on the title we will therefore fit the model with this one


rfModel_Survived.fit(train.loc[:, survival_features_replace], train.loc[:, 'Survived'])

test.loc[:, "Survived"] = rfModel_Survived.predict(test.loc[:, survival_features_replace]).astype(int)
rf_submission = test.loc[:, ["PassengerId", "Survived"]]

# Export to CSV
rf_submission.to_csv("titanic_submissions/Random_fores.csv", index=False)