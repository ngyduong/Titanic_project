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
    if x in ["Mr", 'Mrs', "Miss", "Master"]:
        return(x)
    elif x in ["Don", "Lady", "Sir", "the Countess", "Dona", "Jonkheer"]:
        return("Nobility")
    elif x in ["Rev", "Major", "Col", "Capt", "Dr"]:
        return("Officer")
    elif x == "Mme":
        return("Mrs")
    elif x in ["Ms", "Mlle"]:
        return("Miss")
    else:
        return("ERROR")

titanic["Title"] = titanic.Title.apply(lambda x: generalized_title(x))

# We now have narrowed down the 18 different title into only 6 generalized Title
# As we need the data in categorical form we will create dummy variables from Title

titanic = pd.concat([titanic, pd.get_dummies(titanic["Title"])], axis=1)

# //-- CREATES FAMSIZE \\-- #

# We create the variable Famsize as family size, the size of each family on board
# It is important to add +1 because we have to count the person itself as member of the family
titanic["Famsize"] = titanic['SibSp'] + titanic['Parch'] + 1

# I will  replace Famsize by it's categorical variable as
# Famsize = solo if the person has no family on board and is travelling alone
# Famsize = small_family if the person has 2, 3 or 4 members of his family on board (parents/children/siblings/spouses)
# Famsize = big_family if the person has strictly more than 4 members on his family on board

def Famsize_categorical(x):
    if x == 1:
        return("solo")
    if x in [2,3,4]:
        return ("small_family")
    elif x > 4:
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
        if len(Deck) > 1:
            return(Deck[0])
        else:
            return(Deck)
# This function check if the value is null, if it's true then the function return the null value
# If it's false then the function extract only unique characters from strings and return the upper case value

titanic["Deck"] = titanic["Cabin"].apply(lambda x: get_deck(x))

titanic = pd.concat([titanic, pd.get_dummies(titanic["Deck"], prefix="Deck")], axis=1)

titanic = titanic.drop(columns=['Cabin'])

# //-- Processing Ticket variables \\-- #

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda x: x.strip(), ticket)
    ticket = list(filter(lambda x: not x.isdigit(), ticket))
    if len(ticket) > 0:
        return(ticket[0])
    else:
        return('XXX')

# Extracting dummy variables from tickets:
titanic['Ticket'] = titanic['Ticket'].map(cleanTicket)
titanic = pd.concat([titanic, pd.get_dummies(titanic["Ticket"], prefix='Ticket')], axis=1)
titanic.drop('Ticket', inplace=True, axis=1)

# //-- DEALING WITH EMBARKED and FARE MISSING VALUES \\-- #

## Embarked missing values ##

titanic.Embarked.value_counts()

# The big majority of embarkation were from Southampton (S)
# Since there is only 2 missing values we can either decide to remove them or replace them by the most
# common embarkation port which is Southampton. I personally prefer the latter solution

titanic.Embarked.fillna("S", inplace=True)
titanic = pd.concat([titanic, pd.get_dummies(titanic["Embarked"], prefix="Embarked")], axis=1)

## Fare missing values ##

# As there is only 1 missing value for Fare we can replace the missing value by it's median
titanic.Fare = titanic.Fare.fillna(titanic.Fare.mean())

# ==================== DEALING WITH AGE VARIABLE ==================== #

titanic.Age.describe()
# The mean and the median are close to each other with 29.9 and 28 respectively

# First solution: Replacing missing Age value by the median depending on the title
# Second solution: predicting missing Age with a random Forest
# Third solution; predicting missing Age with a SVM

titanic["Age_Randomforest"] = titanic.Age.copy() #Missing Age predicted by random forest
titanic["Age_SVM"] = titanic.Age.copy() #Missing Age predicted by SVM
titanic["Age_replace"] = titanic.Age.copy() #Missing Age replaced by median depending on title

# //-- Replacing Age depending on Title, Sex and Pclass \\-- #

Age_by_title = pd.DataFrame({'mean_age': titanic.groupby(['Title', 'Sex', 'Pclass']).mean().loc[:, "Age"],
                             'median_age': titanic.groupby(['Title', 'Sex', 'Pclass']).median().loc[:, "Age"]})

Age_by_title = Age_by_title.reset_index()

# We now have the mean and the median age for each title as well as the number of missing age for each title
# We will now replace the missing age by the median age depending on the title

for index, i in enumerate(titanic["Age_replace"]):
    if pd.isnull(i) == False:
        titanic.loc[index, "Age_replace"] = i
    else:
        Sex = titanic.loc[index, "Sex"]
        Title = titanic.loc[index, "Title"]
        Pclass = titanic.loc[index, "Pclass"]

        for index2, j in enumerate(Age_by_title["median_age"]):
            if (Age_by_title.loc[index2,'Sex'] == Sex
                    and Age_by_title.loc[index2,'Title'] == Title
                    and Age_by_title.loc[index2,'Pclass'] == Pclass):
                titanic.loc[index, 'Age_replace'] = Age_by_title.loc[index2, "median_age"].astype('int')
            else:
                pass

# //-- Random forest for Age \\-- #

# we will now split the dataset into 2 datasets, the one with no empty age will be used as training data for the model
titanic_WithAge = titanic[pd.isnull(titanic['Age_Randomforest']) == False]
titanic_WithoutAge = titanic[pd.isnull(titanic['Age_Randomforest'])]

# We will use theses variables as independent variables to predict the Age
independent_variables = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'female', 'male', 'Embarked_C', 'Embarked_Q',
                         'Embarked_S','Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer',
                         'big_family', 'small_family', 'solo', 'Parch', "SibSp", 'Fare',
                         'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F',
                         'Deck_G', 'Deck_T', 'Deck_Unknown']

rfModel_Age = RandomForestRegressor(n_estimators = 2000,
                                    min_samples_split=5,
                                    min_samples_leaf=4,
                                    max_features="sqrt",
                                    max_depth=10,
                                    bootstrap=False,
                                    random_state=1234)

age_accuracies = cross_val_score(estimator=rfModel_Age,
                                 X=titanic_WithAge.loc[:, independent_variables],
                                 y=titanic_WithAge.loc[:, 'Age_Randomforest'],
                                 cv=10,
                                 n_jobs=2)

print("The MEAN CV score is", round(age_accuracies.mean(), ndigits=4))
print("The standard deviation is", round(age_accuracies.std(), ndigits=4))
# The MEAN CV score is 0.4384
# The standard deviation is 0.0542

# Fit the tunned model in the dataset

rfModel_Age.fit(titanic_WithAge.loc[:, independent_variables], titanic_WithAge.loc[:, 'Age_Randomforest'])

titanic_WithoutAge.loc[:, 'Age_Randomforest'] = rfModel_Age.predict(X = titanic_WithoutAge.loc[:, independent_variables]).astype(int)
titanic = titanic_WithAge.append(titanic_WithoutAge).sort_values(by=['PassengerId']).reset_index(drop=True)

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
    elif x <= 30:
        return("17_30")
    elif x <= 40:
        return("31_40")
    else:
        return("over_40")

titanic["Age_group"] = titanic.Age_replace.apply(lambda x: Age_categorical(x))

# ==================== SEPARATE THE DATA AGAIN AND GET BACK OUR TRAIN/TEST DATASETS ==================== #

Age_compare = titanic.loc[titanic.Age.isnull(), ["Age_Randomforest", "Age_SVM", "Age_replace"]]

titanic = titanic.drop(columns="Age")

# I separate the titanic dataframe to their original train/test set
Clean_train = titanic.loc[titanic.Train_set == 1, :].reset_index(drop=True).drop("Train_set", axis=1)
Clean_test = titanic.loc[titanic.Train_set == 0, :].reset_index(drop=True).drop(["Train_set"], axis=1)

# I export them as csv file
Clean_train.to_csv("titanic_data/clean_data/Clean_train.csv", index=False)
Clean_test.to_csv("titanic_data/clean_data/Clean_test.csv", index=False)
titanic.to_csv('titanic_data/clean_data/Clean_titanic.csv', index=False)