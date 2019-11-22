# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import re

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RandomizedSearchCV

# ==================== DATA MANIPULATION ==================== #

test = pd.read_csv("titanic_data/raw_data/test.csv")
train = pd.read_csv("titanic_data/raw_data/train.csv")

# create dummy variables to separate data later on
# if Train_set = 1 the observation was in the train originally
train["Train_set"] = 1
test["Train_set"] = 0

# merge train and test
titanic = train.append(test, ignore_index=True, sort=False)

# ==================== DATA PROCESSING ==================== #

# //-- CREATES DUMMY FOR SEX AND PCLASS VARIABLES \\-- #

titanic = pd.concat([titanic, pd.get_dummies(titanic["Sex"])], axis=1)

titanic = pd.concat([titanic, pd.get_dummies(titanic["Pclass"])], axis=1)
titanic.rename(columns={1:'Pclass_1', 2:'Pclass_2', 3:'Pclass_3'}, inplace=True)

# //-- EXTRACT TITLES FROM NAME VARIABLES \\-- #

comma_split = titanic.Name.str.split(", ", n=1, expand=True)
point_split = comma_split.iloc[:, 1].str.split('.', n=1, expand=True)

titanic["Title"] = point_split.iloc[:, 0]

# We now have the title of each passenger separated from the variable Name
# There is in total 18 different title, I will narrow them down to make a generalized title class

def generalized_title(x):
    if x in ["Mr", 'Mrs', "Miss", "Master", "Dr"]:
        return(x)
    elif x in ["Don", "Lady", "Sir", "the Countess", "Dona", "Jonkheer"]:
        return("Nobility")
    elif x in ["Rev", "Major", "Col", "Capt"]:
        return("Officer")
    elif x == "Mme":
        return("Mrs")
    elif x in ["Ms", "Mlle"]:
        return("Miss")
    else:
        return("ERROR")

titanic["Title"] = titanic.Title.apply(lambda x: generalized_title(x))

# We now have narrowed down the 18 different title into only 7 generalized Title
# As we need the data in categorical form we will create dummy variables from Title

titanic = pd.concat([titanic, pd.get_dummies(titanic["Title"])], axis=1)

# //-- CREATES FAMSIZE \\-- #

# We create the variable Famsize as family size, the size of each family on board
# It is important to add +1 because we have to count the person itself as member of the family
titanic["Famsize"] = titanic['SibSp'] + titanic['Parch'] + 1

# I will  replace Famsize by it's categorical variable as
# Famsize = solo if the person has no family on board and is travelling alone
# Famsize = small_family if the person has 2 or 3 members of his family on board (parents/children/siblings/spouses)
# Famsize = big_family if the person has strictly more than 4 members on his family on board

def Famsize_categorical(x):
    if x == 1:
        return("solo")
    if x in [2,3]:
        return ("small_family")
    elif x > 3:
        return("big_family")
    else:
        return("ERROR")

titanic["Famsize"] = titanic.Famsize.apply(lambda x: Famsize_categorical(x))
titanic = pd.concat([titanic, pd.get_dummies(titanic["Famsize"])], axis=1)

# //-- get Deck variable from Cabin \\-- #

def get_deck(x):
    if pd.isnull(x) == True :
        return("Unknown")
    else:
        deck = re.sub('[^a-zA-Z]+', '', x)
        Deck = "".join(sorted(set(deck), key=deck.index))
        return(Deck.upper())
# This function check if the value is null, if it's true then the function return the null value
# If it's false then the function extract only unique characters from strings and return the upper case value

titanic["Deck"] = titanic["Cabin"].apply(lambda x: get_deck(x))

# //-- Dropping variables \\-- #

titanic = titanic.drop(columns=['Ticket', 'Cabin'])

# //-- DEALING WITH EMBARKED and FARE MISSING VALUES \\-- #

## Embarked missing values ##

titanic.Embarked.value_counts()

# The big majority of embarkation were from Southampton (S)
# Since there is only 2 missing values we can either decide to remove them or replace them by the most
# common embarkation port which is Southampton. I personally prefer the latter solution

titanic.Embarked.fillna("S", inplace=True)
titanic = pd.concat([titanic, pd.get_dummies(titanic["Embarked"])], axis=1)

## Fare missing values ##

# As there is only 1 missing value for Fare we can replace the missing value by it's median
titanic.Fare.fillna(titanic.Fare.median(), inplace=True)

# ==================== DEALING WITH AGE VARIABLE ==================== #

titanic.Age.describe()
# The mean and the median are close to each other with 29.9 and 28 respectively

# First solution: Replacing missing Age value by the median depending on the title
# Second solution: predicting missing Age with a random Forest
# Third solution; predicting missing Age with a SVM

titanic["Age_Randomforest"] = titanic.Age.copy() #Missing Age predicted by random forest
titanic["Age_SVM"] = titanic.Age.copy() #Missing Age predicted by SVM
titanic["Age_replace"] = titanic.Age.copy() #Missing Age replaced by median depending on title

# //-- Replacing Age depending on title \\-- #

Age_by_title = pd.DataFrame({'mean_age': titanic.groupby('Title').mean().loc[:, "Age"],
                             'median_age': titanic.groupby('Title').median().loc[:, "Age"],
                             'count': titanic.Title.value_counts(),
                             'age_missing': titanic.Age.isnull().groupby(titanic['Title']).sum()})

# We now have the mean and the median age for each title as well as the number of missing age for each title
# We will now replace the missing age by the median age depending on the title

for index, i in enumerate(titanic["Age_replace"]):
    if pd.isnull(i) == False:
        titanic.loc[index, "Age_replace"] = i
    else:
        i_title = titanic.loc[index, "Title"]
        titanic.loc[index, 'Age_replace'] = Age_by_title.loc[i_title, "median_age"]

# //-- Random forest for Age \\-- #

# we will now split the dataset into 2 datasets, the one with no empty age will be used as training data for the model
titanic_WithAge = titanic[pd.isnull(titanic['Age_Randomforest']) == False]
titanic_WithoutAge = titanic[pd.isnull(titanic['Age_Randomforest'])]

# We will use theses variables as independent variables to predict the Age
independent_variables = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'female', 'male', 'C', 'Q', 'S',
                         'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer',
                         'big_family', 'small_family', 'solo', 'Parch', "SibSp", 'Fare']

rfModel_Age = RandomForestRegressor()

age_accuracies = cross_val_score(estimator=rfModel_Age,
                                 X=titanic_WithAge.loc[:, independent_variables],
                                 y=titanic_WithAge.loc[:, 'Age_Randomforest'],
                                 cv=10,
                                 n_jobs=2)

print("The MEAN CV score is", round(age_accuracies.mean(), ndigits=2))
print("The standard deviation is", round(age_accuracies.std(), ndigits=2))
# The MEAN CV score is 0.36
# The standard deviation is 0.08

# Fit the tunned model in the dataset

rfModel_Age.fit(titanic_WithAge.loc[:, independent_variables], titanic_WithAge.loc[:, 'Age_Randomforest'])

titanic_WithoutAge.loc[:, 'Age_Randomforest'] = rfModel_Age.predict(X = titanic_WithoutAge.loc[:, independent_variables]).astype(int)
titanic = titanic_WithAge.append(titanic_WithoutAge).sort_values(by=['PassengerId']).reset_index(drop=True)


# =======================================================================================
# # //--  Hyperparameter tuning for Age Random forest  \\-- #
#
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
#
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
#
# rf_random = RandomizedSearchCV(estimator = rfModel_Age,
#                                param_distributions = random_grid,
#                                n_iter = 100,
#                                cv = 10,
#                                verbose=2,
#                                random_state=1234,
#                                n_jobs = 2)
#
# rf_random.fit(titanic_WithAge.loc[:, independent_variables], titanic_WithAge.loc[:, 'Age'])
#
# rf_random.best_params_
#
# Tunned_rfModel_Age = RandomForestRegressor(n_estimators = 2000,
#                                            min_samples_split=5,
#                                            min_samples_leaf=4,
#                                            max_features="sqrt",
#                                            max_depth=10,
#                                            bootstrap=False,
#                                            random_state=1234)
#
# Tunned_age_accuracies = cross_val_score(estimator=Tunned_rfModel_Age,
#                                         X=titanic_WithAge.loc[:, independent_variables],
#                                         y=titanic_WithAge.loc[:, 'Age'],
#                                         cv=10,
#                                         n_jobs=2)
#
# print("The new MEAN CV score is", round(Tunned_age_accuracies.mean(), ndigits=2))
# print("The new standard deviation is", round(Tunned_age_accuracies.std(), ndigits=2))
#
# # The MEAN CV score is 0.45
# # The standard deviation is 0.06
# # The model has increase by 8% and the standard deviation has decreased by 3%
# =======================================================================================


# //-- SVM for Age \\-- #

plt.hist(titanic.Age)
plt.hist(titanic.Age, range=(0, 30))
plt.hist(titanic.Age, range=(30, 80))
# the distribution of the age looks almost like a normal distribution

titanic_WithAge_SVM = titanic[pd.isnull(titanic['Age']) == False]
titanic_WithoutAge_SVM = titanic[pd.isnull(titanic['Age'])]

SVM_reg = SVR(kernel="linear")

age_accuracies_SVM = cross_val_score(estimator=SVM_reg,
                                     X=titanic_WithAge_SVM.loc[:, independent_variables],
                                     y=titanic_WithAge_SVM.loc[:, 'Age_SVM'],
                                     cv=10,
                                     n_jobs=2)

print("The MEAN CV score is", round(age_accuracies_SVM.mean(), ndigits=2))
print("The standard deviation is", round(age_accuracies_SVM.std(), ndigits=2))
# The MEAN CV score is 0.39
# The standard deviation is 0.06

SVM_reg.fit(titanic_WithAge_SVM.loc[:, independent_variables], titanic_WithAge_SVM.loc[:, 'Age_SVM'])

titanic_WithoutAge_SVM.loc[:, 'Age_SVM'] = SVM_reg.predict(X = titanic_WithoutAge_SVM.loc[:, independent_variables]).astype(int)
titanic = titanic_WithAge_SVM.append(titanic_WithoutAge_SVM).sort_values(by=['PassengerId']).reset_index(drop=True)

# //-- Grouped Age using Age_median \\-- #

# Given the shape of the distribution we can separate the Age by group such as
# Age_group = 0_16 if the age is between 0 and 16 included
# Age_group = 17_24 if the age is between 17 and 24 included
# Age_group = 25_30 if the age is between 25 and 30 included
# Age_group = 31_40 if the age is between 31 and 40 included
# Age_group = over_40 if the age is strictly higher than 40

def Age_categorical(x):
    if x <= 16:
        return("0_16")
    elif x <= 24:
        return("17_24")
    elif x <= 30:
        return("25_30")
    elif x <= 40:
        return("31_40")
    else:
        return("over_40")

titanic["Age_group"] = titanic.Age_replace.apply(lambda x: Age_categorical(x))

# ==================== SEPARATE THE DATA AGAIN AND GET BACK OUR TRAIN/TEST DATASETS ==================== #

Age_compare = titanic.loc[:,["Age", "Age_Randomforest", "Age_SVM", "Age_replace"]]

# I separate the titanic dataframe to their original train/test set
Clean_train = titanic.loc[titanic.Train_set == 1, :].reset_index(drop=True).drop("Train_set", axis=1)
Clean_test = titanic.loc[titanic.Train_set == 0, :].reset_index(drop=True).drop(["Train_set"], axis=1)

# I export them as csv file
Clean_train.to_csv("titanic_data/clean_data/Clean_train.csv", index=False)
Clean_test.to_csv("titanic_data/clean_data/Clean_test.csv", index=False)
titanic.to_csv('titanic_data/clean_data/Clean_titanic.csv', index=False)