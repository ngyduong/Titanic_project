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

# ==================== Age selection ==================== #

# age_selection = ['Age_SVM', 'Age_replace', 'Age_Randomforest']
#
# clf = RandomForestClassifier(n_estimators=2000, max_features='sqrt', bootstrap=False)
# clf = clf.fit(train.loc[:, age_selection], train.loc[:, 'Survived'])
#
# features = pd.DataFrame()
# features['Age'] = train.loc[:, age_selection].columns
# features['importance'] = clf.feature_importances_
# features.sort_values(by=['importance'], ascending=True, inplace=True)
# features.set_index('Age', inplace=True)
#
# features.plot(kind='barh', figsize=(20, 10), fontsize=10)

# Age RandomForest est le plus important parmi les 3, nous allons donc utiliser age random forest

# ==================== RANDOM FOREST ==================== #

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

rfModel_Survived = RandomForestClassifier(n_estimators = 1000,
                                          min_samples_split = 5,
                                          min_samples_leaf = 4,
                                          max_features = 'sqrt',
                                          max_depth = 10,
                                          bootstrap = True,
                                          random_state=1234)

# //--  CVS with Age predicted by random forest  \\-- #

randomf_age_rf = cross_val_score(estimator=rfModel_Survived,
                                 X=train.loc[:, survival_features_rf],
                                 y=train.loc[:, 'Survived'],
                                 cv=10,
                                 n_jobs=2)

print("The MEAN CV score is", round(randomf_age_rf.mean(), ndigits=4))
print("The standard deviation is", round(randomf_age_rf.std(), ndigits=4))

# //--  FIT THE MODEMS with age Rf \\-- #

rfModel_Survived.fit(train.loc[:, survival_features_rf], train.loc[:, 'Survived'])

test.loc[:, "Survived"] = rfModel_Survived.predict(test.loc[:, survival_features_rf]).astype(int)
rf_submission = test.loc[:, ["PassengerId", "Survived"]]

# Export to CSV
rf_submission.to_csv("titanic_submissions/survival_prediction_randomforest.csv", index=False)

# Kaggle score: 0.80382