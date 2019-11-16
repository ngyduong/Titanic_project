# ==================== PACKAGES ==================== #

import pandas as pd
# import numpy as np
# import re
import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn


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

Fare_by_title = pd.DataFrame({'mean_fare': titanic.groupby('Title').mean().loc[:, "Fare"],
                              'median_fare': titanic.groupby('Title').median().loc[:, "Fare"],
                              'count': titanic.Title.value_counts(),
                              'fare_missing': titanic.Fare.isnull().groupby(titanic['Title']).sum()})

# As there is only 1 missing value from Fare, and by looking at this dataframe
# we can see that the only missing Fare is from a passenger with the title "Mr"
# I will therefore replace the missing value with the median value of Fare for "Mr"

titanic.Fare.fillna(Fare_by_title.loc['Mr', "median_fare"], inplace=True)

## Dropping the variable Cabin ##

# As there is too many missing value for Cabin, 1014 missing values over 1309 it's not wise to keep the
# variable as it can create noises in the prediction. Therefore I have decided to remove the variable from the dataset
titanic = titanic.drop(columns="Cabin")


# ==================== Creating derived variables ==================== #


# //-- Derived variables from SibSp and Parch \\-- #

## We create the variable Famsize as family size, the size of each family on board
titanic.Famsize = titanic['SibSp'] + titanic['Parch'] + 1
# It is important to add +1 because we have to count the person itself as member of the family

titanic.Famsize.value_counts()
titanic.Famsize.hist()
# Given the distribution of the variable we can create categorical variables based on group of family size

# I will therefore replace Famsize by it's categorical variable as :
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

# //-- Derived variables from Age \\-- #

plt.hist(titanic.Age)
plt.hist(titanic.Age, range=(0, 30))
plt.hist(titanic.Age, range=(30, 80))
titanic.Age.describe()
# the distribution of the age looks almost like a normal distribution

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

titanic["Age_group"] = titanic.Age.apply(lambda x: Age_categorical(x))

titanic.Age_group.value_counts()
titanic.Age_group.hist()




