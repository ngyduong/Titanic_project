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

# ==================== RANDOM FOREST ==================== #

rfModel_Survived = RandomForestClassifier(n_estimators = 1000,
                                          min_samples_split = 5,
                                          min_samples_leaf = 4,
                                          max_features = 'sqrt',
                                          max_depth = 10,
                                          bootstrap = True,
                                          random_state=1234)

# //--  CVS with Age predicted by random forest  \\-- #

age_rf = cross_val_score(estimator=rfModel_Survived,
                         X=train.loc[:, survival_features_rf],
                         y=train.loc[:, 'Survived'],
                         cv=10,
                         n_jobs=2)

print("The MEAN CV score is", round(age_rf.mean(), ndigits=2))
print("The standard deviation is", round(age_rf.std(), ndigits=2))
# The MEAN CV score is 0.84
# The standard deviation is 0.04

# //--  CVS with Age predicted by SVM  \\-- #

age_svm = cross_val_score(estimator=rfModel_Survived,
                          X=train.loc[:, survival_features_svm],
                          y=train.loc[:, 'Survived'],
                          cv=10,
                          n_jobs=2)

print("The MEAN CV score is", round(age_svm.mean(), ndigits=2))
print("The standard deviation is", round(age_svm.std(), ndigits=2))
# The MEAN CV score is 0.84
# The standard deviation is 0.04

# //--  CVS with Age replaced by median depending on title  \\-- #

age_replace = cross_val_score(estimator=rfModel_Survived,
                              X=train.loc[:, survival_features_replace],
                              y=train.loc[:, 'Survived'],
                              cv=10,
                              n_jobs=2)

print("The MEAN CV score is", round(age_replace.mean(), ndigits=2))
print("The standard deviation is", round(age_replace.std(), ndigits=2))
# The MEAN CV score is 0.84
# The standard deviation is 0.04

# //--  FIT THE MODEMS with age Rf \\-- #

rfModel_Survived.fit(train.loc[:, survival_features_rf], train.loc[:, 'Survived'])

test.loc[:, "Survived"] = rfModel_Survived.predict(test.loc[:, survival_features_rf]).astype(int)
rf_submission = test.loc[:, ["PassengerId", "Survived"]]

# Export to CSV
rf_submission.to_csv("titanic_submissions/survival_prediction_randomforest.csv", index=False)