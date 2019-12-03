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

reduced_features2 = ['Ticket_SCA4', 'Ticket_Fa', 'Ticket_AS', 'Ticket_CASOTON', 'Ticket_SCOW',
                     'Ticket_SC', 'Ticket_SOTONO2', 'Ticket_PPP', 'Ticket_SP', 'Ticket_SOP',
                     'Ticket_SCAH', 'Ticket_SCParis', 'Ticket_FC', 'Ticket_FCC', 'Ticket_A4',
                     'Ticket_SOC', 'Ticket_WEP', 'Ticket_PP', 'Nobility', 'Ticket_SCPARIS',
                     'Ticket_LINE', 'Deck_G', 'Deck_F', 'Ticket_SOPP', 'Ticket_STONO2', 'Ticket_SOTONOQ',
                     'Ticket_C', 'Deck_A', 'Ticket_WC', 'Ticket_CA', 'Ticket_A5', 'Embarked_Q',
                     'Embarked_Q', 'Officer', 'Ticket_PC', 'Ticket_SWPP', 'Ticket_STONO', 'Embarked_C',
                     'Deck_B', 'Embarked_C', 'Deck_C', 'Embarked_S', 'Deck_D', 'Embarked_S', 'Deck_E',
                     'Master', 'solo', 'Ticket_XXX', 'small_family', 'Pclass_2', 'Pclass_1', 'big_family',
                     'Parch', 'Mrs', 'Deck_Unknown', 'SibSp', 'Miss', 'Pclass_3', 'male', 'female', 'Mr',
                     'Fare', 'Age_Randomforest']

# ==================== Support Vector Machine (SVM) ==================== #

svclassifier = SVC(kernel="linear", random_state=1234)

svc_age_rf = cross_val_score(estimator=svclassifier,
                             X=train.loc[:, reduced_features2],
                             y=train.loc[:, 'Survived'],
                             cv=5,
                             n_jobs=2)

print("The MEAN CV score is", round(svc_age_rf.mean(), ndigits=4))
print("The standard deviation is", round(svc_age_rf.std(), ndigits=4))
# The MEAN CV score is 0.8204
# The standard deviation is 0.0217

## Fit the model

svclassifier.fit(train.loc[:, reduced_features2], train.loc[:, 'Survived'])
test.loc[:, "Survived"] = svclassifier.predict(test.loc[:, reduced_features2]).astype(int)
SVM_test_basic = test.loc[:, ["PassengerId", "Survived"]]
SVM_test_basic.to_csv("titanic_submissions/survival_prediction_svm.csv", index=False)