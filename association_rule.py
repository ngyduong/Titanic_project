# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #

import pandas as pd

# ==================== DATA MANIPULATION ==================== #

titanic = pd.read_csv("titanic_data/clean_data/Clean_titanic.csv")
titanic = titanic.drop(['PassengerId', 'Name', 'Age', 'Train_set', 'female', 'male', 'Pclass_1',
                        'Pclass_2', 'Pclass_3', 'Dr', 'Master', 'Miss', 'Mr', 'Mrs','Nobility',
                        'Officer', 'big_family', 'small_family', 'solo', 'Deck', 'Age_Randomforest',
                        'Age_SVM', 'Age_replace', "C", "Q", "S", "Fare"], axis=1)

# ==================== DATA PROCESSING ==================== #

# //-- Pclass \\-- #

def Pclass_process(x):
    if x == 1:
        return('First_class')
    elif x == 2:
        return('Second_class')
    else:
        return('Third_class')

titanic.loc[:, "Pclass"] = titanic.Pclass.apply(lambda x: Pclass_process(x))

# //-- Survived \\-- #

def Survived_process(x):
    if x == 1:
        return("Survived")
    else:
        return("Died")

titanic.loc[:, "Survived"] = titanic.Survived.apply(lambda x: Survived_process(x))

# //-- SibSp (# of siblings / spouses aboard the Titanic) \\-- #

titanic.SibSp.value_counts()

def SibSp_process(x):
    if x == 0:
        return("no siblings/spouses aboard")
    elif x == 1:
        return("1 siblings/spouses aboard")
    else:
        return("2 or more siblings/spouses aboard")

titanic.loc[:, "SibSp"] = titanic.SibSp.apply(lambda x: SibSp_process(x))

# //-- Parch (# of parents / children aboard the Titanic) \\-- #

titanic.Parch.value_counts()

def Parch_process(x):
    if x == 0:
        return("no parents/childrien aboard")
    elif x == 1:
        return("1 parents/children aboard")
    else:
        return('2 or more parents/children aboard')

titanic.loc[:, "Parch"] = titanic.Parch.apply(lambda x: Parch_process(x))

# ==================== ASSOCIATION RULE ==================== #

