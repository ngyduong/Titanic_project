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

survival_features_rf = ['SibSp', 'Parch', 'Fare', 'female', 'male', 'Pclass_1', 'Pclass_2', 'Pclass_3',
                        'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                        'small_family', 'solo', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_Randomforest',
                        'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F',
                        'Deck_FE', 'Deck_FG', 'Deck_G', 'Unknown']

survival_features_svm = ['SibSp', 'Parch', 'Fare', 'female', 'male','Pclass_1', 'Pclass_2', 'Pclass_3',
                        'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                        'small_family', 'solo', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_SVM',
                        'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F',
                        'Deck_FE', 'Deck_FG', 'Deck_G', 'Unknown']

survival_features_replace = ['SibSp', 'Parch', 'Fare', 'female', 'male','Pclass_1', 'Pclass_2', 'Pclass_3',
                            'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer', 'big_family',
                            'small_family', 'solo', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_replace',
                            'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F',
                            'Deck_FE', 'Deck_FG', 'Deck_G', 'Unknown']

# ==================== Support Vector Machine (SVM) ==================== #

svclassifier = SVC(kernel="linear",
                   random_state=1234)

# //--  CVS with Age predicted by random forest  \\-- #

svc_age_rf = cross_val_score(estimator=svclassifier,
                             X=train.loc[:, survival_features_rf],
                             y=train.loc[:, 'Survived'],
                             cv=10,
                             n_jobs=2)

print("The MEAN CV score is", round(svc_age_rf.mean(), ndigits=4))
print("The standard deviation is", round(svc_age_rf.std(), ndigits=4))
# The MEAN CV score is 0.8204
# The standard deviation is 0.0217

# //--  CVS with Age predicted by SVM classifier  \\-- #

svc_age_svm = cross_val_score(estimator=svclassifier,
                              X=train.loc[:, survival_features_svm],
                              y=train.loc[:, 'Survived'],
                              cv=10,
                              n_jobs=2)

print("The MEAN CV score is", round(svc_age_svm.mean(), ndigits=4))
print("The standard deviation is", round(svc_age_svm.std(), ndigits=4))
# The MEAN CV score is 0.8193
# The standard deviation is 0.0214

# //--  CVS with Age replaced by median depending on title  \\-- #

svc_age_replace = cross_val_score(estimator=svclassifier,
                                  X=train.loc[:, survival_features_replace],
                                  y=train.loc[:, 'Survived'],
                                  cv=10,
                                  n_jobs=2)

print("The MEAN CV score is", round(svc_age_replace.mean(), ndigits=4))
print("The standard deviation is", round(svc_age_replace.std(), ndigits=4))
# The MEAN CV score is 0.8227
# The standard deviation is 0.0226

## Fit the model

svclassifier.fit(train.loc[:, survival_features_replace], train.loc[:, 'Survived'])
test.loc[:, "Survived"] = svclassifier.predict(test.loc[:, survival_features_replace]).astype(int)
SVM_test_basic = test.loc[:, ["PassengerId", "Survived"]]
SVM_test_basic.to_csv("titanic_submissions/survival_prediction_svm222.csv", index=False)