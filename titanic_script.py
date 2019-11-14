# ==================== PACKAGES ==================== #

import pandas as pd
import numpy as np
import sklearn
import re
import math

# ==================== IMPORT DATA ==================== #

test = pd.read_csv("titanic_data/test.csv")
train = pd.read_csv("titanic_data/train.csv")

# ==================== CLEANING DATA ==================== #

##  We create the variable Famsize as family size, the size of each family on board
train['Famsize'] = train['SibSp'] + train['Parch']

# # We convert all observations from Cabin variable into string
# train["Cabin"] = train["Cabin"].apply(lambda x: str(x))

#  We extract the Deck from cabin number and name the new variable Deck

def get_deck(x):
    if pd.isnull(x) == True :
        return(x)
    else:
        str_x = str(x)
        deck = re.sub('[^a-zA-Z]+', '', str_x)
        Deck = "".join(sorted(set(deck), key=deck.index))
        return(Deck.upper())

train["Deck"] = train["Cabin"].apply(lambda x: get_deck(x))

