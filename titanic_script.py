# ==================== PACKAGES ==================== #


import pandas as pd
import numpy as np
# import re
import matplotlib.pyplot as plt
# import seaborn as sns
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# ==================== DATA MANIPULATION ==================== #


test = pd.read_csv("titanic_data/test.csv")
train = pd.read_csv("titanic_data/train.csv")

# create dummy variables to separate data later on
# if Train_set = 1 the observation was in the train originally
train["Train_set"] = 1
test["Train_set"] = 0

# merge train and test
titanic = train.append(test, ignore_index=True, sort=False)


# ==================== DATA PROCESSING ==================== #


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

# As we need the data in categorical form we will create dummy variables from Famsize
titanic = pd.concat([titanic, pd.get_dummies(titanic["Famsize"])], axis=1)

# //-- CREATES DUMMY FOR SEX AND EMBARKED VARIABLES \\-- #

titanic = pd.concat([titanic, pd.get_dummies(titanic["Sex"])], axis=1)
titanic = pd.concat([titanic, pd.get_dummies(titanic["Embarked"])], axis=1)

# ==================== DEALING WITH EMBARKED AND FARE MISSING VALUES ==================== #


titanic.isnull().sum()
# It seems that there is mostly missing values for the variables Embarked, age, Fare and Deck
# There is 418 missing value for survived because in the original test data set there was no variable survived

# //-- Dropping variables \\-- #

# As there is too many missing value for Cabin, 1014 missing values over 1309 it's not wise to keep the
# variable as it can create noises in the prediction. Therefore I have decided to remove the variable from the dataset
# There doesn't seem to have any valuable information in the variable Ticket so I will drop the variable off as well
titanic = titanic.drop(columns=["Cabin", 'Ticket'])

# //-- Let's try now to fill the missing values for each variables \\-- #

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


# ==================== DEALING WITH AGE MISSING VALUES ==================== #

# Let's have a first statistical analysis

titanic.Age.describe()
titanic.Age.hist()
# The mean and the median are close to each other with 29.9 and 28 respectively
# The easy solution would be to replace the missing ages by the mean or median
# but we can also use random forest to predict the age

# we will now split the dataset into 2 datasets, the one with no empty age will be used as training data for the model
titanic_WithAge = titanic[pd.isnull(titanic['Age']) == False]
titanic_WithoutAge = titanic[pd.isnull(titanic['Age'])]

# We will use theses variables as independent variables to predict the Age
independent_variables = ['Pclass', 'female', 'male', 'C', 'Q', 'S',
                         'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer',
                         'big_family', 'small_family', 'solo',
                         'Parch', "SibSp",
                         # 'Fare'
                         ]

# //-- First Random forest estimator \\-- #

rfModel_Age = RandomForestRegressor(n_estimators=750, random_state=1234)

age_accuracies = cross_val_score(estimator=rfModel_Age,
                                 X=titanic_WithAge.loc[:, independent_variables],
                                 y=titanic_WithAge.loc[:, 'Age'],
                                 cv=10)

print("The MEAN CV accuracy of Age prediction is", round(age_accuracies.mean(), ndigits=2))
print("The MAX CV accuracy of Age prediction is", round(age_accuracies.max(), ndigits=2))
print("The MIN CV accuracy of Age prediction is", round(age_accuracies.min(), ndigits=2))
# The accuracy is not too bad but we can maybe improve the score by performing a hyperparameter tuning

print(rfModel_Age.get_params())

# //--  Hyperparameter tuning for Age Random forest  \\-- #

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rfModel_Age2 = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rfModel_Age2,
                               param_distributions = random_grid,
                               n_iter = 20,
                               cv = 10,
                               verbose=2,
                               random_state=1234,
                               n_jobs = 2)

rf_random.fit(titanic_WithAge.loc[:, independent_variables], titanic_WithAge.loc[:, 'Age'])

rf_random.best_params_

# {   'n_estimators': 1800,
#     'min_samples_split': 10,
#     'min_samples_leaf': 2,
#     'max_features': 'auto',
#     'max_depth': 80,
#     'bootstrap': True   }

Tunned_rfModel_Age = RandomForestRegressor(n_estimators = 1800,
                                           min_samples_split=10,
                                           min_samples_leaf=2,
                                           max_features="auto",
                                           max_depth=80,
                                           bootstrap=True,
                                           random_state=1234)

Tunned_age_accuracies = cross_val_score(estimator=Tunned_rfModel_Age,
                                        X=titanic_WithAge.loc[:, independent_variables],
                                        y=titanic_WithAge.loc[:, 'Age'],
                                        cv=5)

print("The MEAN tunned CV accuracy of Age prediction is", round(Tunned_age_accuracies.mean(), ndigits=2))
print("The MAX tunned CV accuracy of Age prediction is", round(Tunned_age_accuracies.max(), ndigits=2))
print("The MIN tunned CV accuracy of Age prediction is", round(Tunned_age_accuracies.min(), ndigits=2))

# The model has slightly improved, by roughly 1%.

# //--  Fit the tunned model in the dataset  \\-- #

Tunned_rfModel_Age.fit(titanic_WithAge.loc[:, independent_variables], titanic_WithAge.loc[:, 'Age'])
Predicted_age = Tunned_rfModel_Age.predict(X = titanic_WithoutAge.loc[:,independent_variables])
titanic_WithoutAge['Age'] = Predicted_age.astype(int)

titanic = titanic_WithAge.append(titanic_WithoutAge)
titanic = titanic.sort_values(by=['PassengerId']).reset_index(drop=True)

# We now have all observations of Age without missing values by random forest prediction

titanic.Age.describe()
plt.hist(titanic.Age)
plt.hist(titanic.Age, range=(0, 30))
plt.hist(titanic.Age, range=(30, 80))
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

titanic = pd.concat([titanic, pd.get_dummies(titanic["Age_group"])], axis=1)