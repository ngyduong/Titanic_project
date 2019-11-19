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

# ==================== RANDOM FOREST ==================== #

rfModel_Survived = RandomForestClassifier()

survived_accuracies = cross_val_score(estimator=rfModel_Survived,
                                      X=train_X.loc[:, survival_features],
                                      y=train_Y.loc[:, 'Survived'],
                                      cv=10,
                                      n_jobs=2)

print("The MEAN CV score is", round(survived_accuracies.mean(), ndigits=2))
print("The standard deviation is", round(survived_accuracies.std(), ndigits=2))

# The MEAN CV score is 0.8
# The standard deviation is 0.05

# //--  Hyperparameter tuning RandomSearch \\-- #

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = rfModel_Survived,
                               param_distributions = random_grid,
                               n_iter = 100,
                               cv = 10,
                               verbose=2,
                               random_state=1234,
                               n_jobs = 3)

rf_random.fit(train_X.loc[:, survival_features], train_Y.loc[:, 'Survived'])

rf_random.best_params_

Tunned_rfModel_Survived = RandomForestClassifier(n_estimators = 200,
                                                 min_samples_split=2,
                                                 min_samples_leaf=4,
                                                 max_features="sqrt",
                                                 max_depth=40,
                                                 bootstrap=True,
                                                 random_state=1234)

survived_accuracies_RS = cross_val_score(estimator=Tunned_rfModel_Survived,
                                         X=train_X.loc[:, survival_features],
                                         y=train_Y.loc[:, 'Survived'],
                                         cv=10,
                                         n_jobs=2)

print("The MEAN CV score is", round(survived_accuracies_RS.mean(), ndigits=2))
print("The standard deviation is", round(survived_accuracies_RS.std(), ndigits=2))

# The MEAN CV score is 0.84
# The standard deviation is 0.04
# The mean accuracy has improved by 3% and the  standard deviation stayed the same

# # //--  Hyperparameter tuning Grid Search \\-- #
#
# # Create the parameter grid based on the results of random search
# param_grid = {'bootstrap': [True],
#               'max_depth': [40, 60, 80, 100],
#               'max_features': ["sqrt", "auto"],
#               'min_samples_split': [2, 3, 5],
#               'min_samples_leaf': [4, 10, 12],
#               'n_estimators': [200, 300, 1000]}
#
# grid_search = GridSearchCV(estimator = rfModel_Survived,
#                            param_grid = param_grid,
#                            cv = 3,
#                            n_jobs = 3,
#                            verbose = 2)
#
# grid_search.fit(train_X.loc[:, survival_features], train_Y.loc[:, 'Survived'])
#
# print("Best score: {}".format(grid_search.best_score_))
# print("Optimal params: {}".format(grid_search.best_estimator_))

# //--  Fit the model to the test set \\-- #

# ResearchGrid
Tunned_rfModel_Survived.fit(train.loc[:, survival_features], train.loc[:, 'Survived'])
test.loc[:, "Survived"] = Tunned_rfModel_Survived.predict(test.loc[:, survival_features]).astype(int)
Random_forest_test_RandomSearch = test.loc[:, ["PassengerId", "Survived"]]

# Re-importe test dataset
test = pd.read_csv("titanic_data/Clean_test.csv")

# # ResearchGrid
# test.loc[:, "Survived"] = grid_search.predict(test.loc[:, survival_features]).astype(int)
# Random_forest_test_GridSearch = test.loc[:, ["PassengerId", "Survived"]]

# Export to CSV
Random_forest_test_RandomSearch.to_csv("titanic_submissions/Random_forest_test.csv", index=False)
# Random_forest_test_GridSearch.to_csv("titanic_submissions/Random_forest_test.csv", index=False)