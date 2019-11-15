# ==================== PACKAGES ==================== #

import pandas as pd
import numpy as np
import re
import matplotlib as plt
import seaborn as sns
import sklearn

# ==================== DATA MANIPULATION ==================== #

test = pd.read_csv("titanic_data/test.csv")
train = pd.read_csv("titanic_data/train.csv")

# create dummy variables to separate data later on
# if Train_set = 1 the observation was in the train originally
train["Train_set"] = 1
test["Train_set"] = 0

# merge train and test
titanic = train.append(test, ignore_index=True, sort=False)

# ==================== CLEANING DATA ==================== #

# //-- New variable Famsize \\-- #

titanic['Famsize'] = titanic['SibSp'] + titanic['Parch'] + 1
# We create the variable Famsize as family size, the size of each family on board
# It is important to add +1 because we have to count the person itself as member of the family

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
# It seems that there is mostly missing values for the variables Embarked, age, Fare and Deck
# There is 418 missing value for survived because in the original test data set there was no variable survived

# //-- Let's try now to fill the missing values for each variables \\-- #

## Fill NaN in Age variable ##

# Let's have a first statistical analysis

titanic.Age.describe()
titanic.Age.hist()
# The mean and the median are close to each other with 29.7 and 28 respectively
# The easy solution would be to replace the missing ages by the mean or median but
# we can also split the name and get the title for each class and then uses the titles
# to get a better approximation of the age

comma_split = titanic.Name.str.split(", ", n=1, expand=True)
point_split = comma_split.iloc[:, 1].str.split('.', n=1, expand=True)

titanic["Title"] = point_split.iloc[:, 0]
# We now have the title of each passenger separated from the variable Name
# There is in total 18 different title, I will narrow them down to make a generalized title class

def generalized_title(x):
    if x in ["Mr", 'Mrs', "Miss", "Master", "Dr"]:
        return(x)
    elif x in ["Don", "Lady", "Sir", "the Countess", "Dona", "Jonkheer", ""]:
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


Age_by_title = pd.DataFrame({'mean_age': titanic.groupby('Title').mean().loc[:, "Age"],
                             'median_age': titanic.groupby('Title').median().loc[:, "Age"],
                             'count': titanic.Title.value_counts(),
                             'age_missing': titanic.Age.isnull().groupby(titanic['Title']).sum()})
# We now have the mean and the median age for each title as well as the number of missing age for each title

# We will now replace the missing age by the median age depending on the title
for index, i in enumerate(titanic["Age"]):
    if pd.isnull(i) == False:
        titanic.loc[index, "Age"] = i
    else:
        i_title = titanic.loc[index, "Title"]
        titanic.loc[index, 'Age'] = Age_by_title.loc[i_title, "median_age"]

## Fill NaN in Embarked variable ##

titanic.Embarked.value_counts()
# The big majority of embarkation were from Southampton (S)
# Since there is only 2 missing values we can either decide to remove them or replace them by the most
# common embarkation port which is Southampton. I personally prefer the latter solution.

# let's replace the missing embarkation port by Southampton (S)
titanic.Embarked.fillna("S", inplace=True)

## Fill NaN in Fare variable ##

titanic.Fare.describe()
titanic.Fare.hist()
# The mean Fare is 33.3 meanwhile the median is 14.45
# Most of the distribution is 0 and 50, let's plot the distribution of Fare only between 0 and 50

fare_filtered = pd.Series(filter(lambda x: x <= 0, titanic.Fare))
fare_filtered.hist()
# Most of the distribution is actually before 15 so I have decided to replace the missing value by it's median

# let's replace the missing fare it's median
titanic.Fare.fillna(titanic.Fare.median(), inplace=True)

## Fill NaN in Deck variable ##

# As there is too many missing values for Deck (and henceforh Cabin) with over 1000 missing values
# it is not wise to replace the value by the most common value or drop all observations with missing values
# Therefore i will replace the missing values by a "Unknown" for unknown deck

titanic.Deck.fillna("Unknown", inplace=True)




