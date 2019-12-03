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

survival_features_rf = ['SibSp', 'Parch',
                       'Fare', 'female', 'male', 'Pclass_1', 'Pclass_2',
                       'Pclass_3', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility',
                       'Officer', 'big_family', 'small_family', 'solo',
                       'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G',
                       'Deck_T', 'Deck_Unknown', 'Ticket_A', 'Ticket_A4', 'Ticket_A5',
                       'Ticket_AQ3', 'Ticket_AQ4', 'Ticket_AS', 'Ticket_C', 'Ticket_CA',
                       'Ticket_CASOTON', 'Ticket_FC', 'Ticket_FCC', 'Ticket_Fa', 'Ticket_LINE',
                       'Ticket_LP', 'Ticket_PC', 'Ticket_PP', 'Ticket_PPP', 'Ticket_SC',
                       'Ticket_SCA3', 'Ticket_SCA4', 'Ticket_SCAH', 'Ticket_SCOW',
                       'Ticket_SCPARIS', 'Ticket_SCParis', 'Ticket_SOC', 'Ticket_SOP',
                       'Ticket_SOPP', 'Ticket_SOTONO2', 'Ticket_SOTONOQ', 'Ticket_SP',
                       'Ticket_STONO', 'Ticket_STONO2', 'Ticket_STONOQ', 'Ticket_SWPP',
                       'Ticket_WC', 'Ticket_WEP', 'Ticket_XXX', 'Embarked_C', 'Embarked_Q',
                       'Embarked_S', 'Age_Randomforest']

# ==================== Support Vector Machine (SVM) ==================== #

svclassifier = SVC(kernel="linear",
                   gamma = "scale",
                   random_state=1234,
                   shrinking=False)

svc_age_rf = cross_val_score(estimator=svclassifier,
                             X=train.loc[:, survival_features_rf],
                             y=train.loc[:, 'Survived'],
                             cv=10,
                             n_jobs=2)

print("The MEAN CV score is", round(svc_age_rf.mean(), ndigits=4))
print("The standard deviation is", round(svc_age_rf.std(), ndigits=4))
# The MEAN CV score is 0.8216
# The standard deviation is 0.0247

## Fit the model

svclassifier.fit(train.loc[:, survival_features_rf], train.loc[:, 'Survived'])
test.loc[:, "Survived"] = svclassifier.predict(test.loc[:, survival_features_rf]).astype(int)
SVM_test_basic = test.loc[:, ["PassengerId", "Survived"]]
SVM_test_basic.to_csv("titanic_submissions/survival_prediction_svm.csv", index=False)

# Kaggle score: 0.77990