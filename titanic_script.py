# ==================== PACKAGES ==================== #

import pandas as pd
import numpy as np
import sklearn
import re

# ==================== IMPORT DATA ==================== #

test = pd.read_csv("titanic_data/test.csv")
train = pd.read_csv("titanic_data/train.csv")

# ==================== CLEANING DATA ==================== #

# //-- We create the variable Famsize as family size, the size of each family on board \\-- #

train['Famsize'] = train['SibSp'] + train['Parch']
test['Famsize'] = test['SibSp'] + test['Parch']

# //-- We extract the Deck from cabin number and name the new variable Deck \\-- #

def get_deck(x):
    if pd.isnull(x) == True :
        return(x)
    else:
        deck = re.sub('[^a-zA-Z]+', '', x)
        Deck = "".join(sorted(set(deck), key=deck.index))
        return(Deck.upper())

# This function check if the value is null, if it's true then the function return the null value
# If it's false then the function extract only unique characters from strings and return the upper case value

train["Deck"] = train["Cabin"].apply(lambda x: get_deck(x))
test["Deck"] = test["Cabin"].apply(lambda x: get_deck(x))

# //-- We check the missing values in each dataset \\-- #

train.isnull().sum()
test.isnull().sum()

# It seems that there is mostly missing values for the variables Cabin, Embarked, age and Deck




