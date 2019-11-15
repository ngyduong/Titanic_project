# ==================== PACKAGES ==================== #

import pandas as pd
import numpy as np
import re
import matplotlib as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import Imputer

# ==================== IMPORT DATA ==================== #

test = pd.read_csv("titanic_data/test.csv")
train = pd.read_csv("titanic_data/train.csv")

# save PassengerId for final submission
passengerId = test.PassengerId

# create dummy variables to separate data later on
# if Train_set = 1 the observation was in the train originally
train["Train_set"] = 1
test["Train_set"] = 0

# merge train and test
titanic = train.append(test, ignore_index=True, sort=False)

# ==================== CLEANING DATA ==================== #

# //-- We create the variable Famsize as family size, the size of each family on board \\-- #

titanic['Famsize'] = titanic['SibSp'] + titanic['Parch']

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

titanic["Deck"] = titanic["Cabin"].apply(lambda x: get_deck(x))

# //-- We check the missing values in each dataset \\-- #

titanic.isnull().sum()
# It seems that there is mostly missing values for the variables Cabin, Embarked, age, Fare and Deck
# There is 418 missing value for survived because in the original test data set there was no variable survived

# //-- Let's try now to fill the missing values for each variables \\-- #

## Fill NaN in Age variable ##

# Let's have a first statistical analysis

titanic.Age.describe()
titanic.Age.hist()
# The mean and the median are close to each other with 29.7 and 28 respectively
# Most of the observations are between 20 and 30 yo
# We can maybe replace the missing values by the median

# We now will replace the missing age by it's median

imp = Imputer(missing_values='NaN', strategy='median', axis=1)
titanic["New_age"] = imp.fit_transform(titanic['Age'].values.reshape(1, -1)).T
# We now have new age as our new age variable without missing values

titanic.New_age.describe()
# The median is still 28 and the mean is slightly smaller, decreased from 29.7 to 29.5

## Fill NaN in Embarked variable ##

titanic.Embarked.value_counts()
# The big majority of embarkation were from Southampton (S)
# Since there is only 2 missing values we can either decide to remove them or replace them by the most
# common embarkation port which is Southampton. I personally prefer the latter solution.

# let's replace the missing embarkation port by Southampton (S)
titanic.Embarked.fillna("S",inplace=True)

## Fill NaN in Fare variable ##

titanic.Fare.describe()
titanic.Fare.hist()
# The mean Fare is 33.3 meanwhile the median is 14.45
# Most of the distribution is 0 and 50, let's plot the distribution of Fare only between 0 and 50

fare_filtered = pd.Series(filter(lambda x: x <= 0, titanic.Fare))
fare_filtered.hist()
# Most of the distribution is actually before 15 so I have decided to replace the missing value by it's median

titanic.Fare.fillna(titanic.Fare.median(),inplace=True)




