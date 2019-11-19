# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
# from sklearn.linear_model import LassoLarsCV

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


# //-- CREATES DUMMY FOR SEX AND PCLASS VARIABLES \\-- #

titanic = pd.concat([titanic, pd.get_dummies(titanic["Sex"])], axis=1)
titanic = pd.concat([titanic, pd.get_dummies(titanic["Pclass"])], axis=1)
titanic.rename(columns={1:'Pclass_1', 2:'Pclass_2', 3:'Pclass_3'}, inplace=True)
titanic = titanic.drop(["Sex", "Pclass"], axis=1)

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
titanic=titanic.drop('Title', axis=1)

# //-- CREATES FAMSIZE \\-- #

# We create the variable Famsize as family size, the size of each family on board
titanic["Famsize"] = titanic['SibSp'] + titanic['Parch'] + 1
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
titanic = titanic.drop('Famsize', axis=1)

# //-- Dropping variables \\-- #

# As there is too many missing value for Cabin, 1014 missing values over 1309 it's not wise to keep the
# variable as it can create noises in the prediction. Therefore I have decided to remove the variable from the dataset
# There doesn't seem to have any valuable information in the variable Ticket so I will drop the variable off as well
titanic = titanic.drop(columns=["Cabin", 'Ticket'])


# ==================== DEALING WITH MISSING VALUES ==================== #


titanic.isnull().sum()
# It seems that there is mostly missing values for the variables Embarked, Age and Fare

# //-- Let's try now to fill the missing values for each variables \\-- #

## Fill NaN in Embarked variable ##

titanic.Embarked.value_counts()

# The big majority of embarkation were from Southampton (S)
# Since there is only 2 missing values we can either decide to remove them or replace them by the most
# common embarkation port which is Southampton. I personally prefer the latter solution.

# let's replace the missing embarkation port by Southampton (S)
titanic.Embarked.fillna("S", inplace=True)

# Now that there is no more missing value we can create dummy variables for Embarked
titanic = pd.concat([titanic, pd.get_dummies(titanic["Embarked"])], axis=1)
titanic = titanic.drop("Embarked", axis=1)

## Fill NaN in Fare variable ##

titanic.Fare.fillna(titanic.Fare.median(), inplace=True)
# As there is only 1 missing value for Fare we can replace the missing value by it's median


# ==================== Fill NaN in Age variable with RANDOM FOREST ==================== #


# Let's have a first statistical analysis

titanic.Age.describe()
# The mean and the median are close to each other with 29.9 and 28 respectively

# we will now split the dataset into 2 datasets, the one with no empty age will be used as training data for the model
titanic_WithAge = titanic[pd.isnull(titanic['Age']) == False]
titanic_WithoutAge = titanic[pd.isnull(titanic['Age'])]

# We will use theses variables as independent variables to predict the Age
independent_variables = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'female', 'male', 'C', 'Q', 'S',
                         'Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Nobility', 'Officer',
                         'big_family', 'small_family', 'solo', 'Parch', "SibSp", 'Fare']


# //-- Features selections by Lasso Regression \\-- #
#
# x_train = titanic_WithAge.loc[:, independent_variables]
# y_train = titanic_WithAge.loc[:, 'Age']
#
# Model_Lasso = LassoLarsCV(cv = 10, precompute = False).fit(x_train, y_train)
#
# lasso_coefs = pd.DataFrame(Model_Lasso.coef_)
# tlasso_coefs['features'] = pd.Series(titanic_WithAge.loc[:, independent_variables].columns)
# tlasso_coefs = tlasso_coefs.sort_values(by=[0])
#
# lasso_coef = list(tlasso_coefs.loc[tlasso_coefs[0] > 0, "features"])


# //-- First Random forest estimator \\-- #

rfModel_Age = RandomForestRegressor()

age_accuracies = cross_val_score(estimator=rfModel_Age,
                                 X=titanic_WithAge.loc[:, independent_variables],
                                 y=titanic_WithAge.loc[:, 'Age'],
                                 cv=10,
                                 n_jobs=2)

print("The MEAN CV score is", round(age_accuracies.mean(), ndigits=2))
print("The standard deviation is", round(age_accuracies.std(), ndigits=2))

# The MEAN CV score is 0.36
# The standard deviation is 0.09

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

rf_random = RandomizedSearchCV(estimator = rfModel_Age,
                               param_distributions = random_grid,
                               n_iter = 100,
                               cv = 10,
                               verbose=2,
                               random_state=1234,
                               n_jobs = 2)

rf_random.fit(titanic_WithAge.loc[:, independent_variables], titanic_WithAge.loc[:, 'Age'])

rf_random.best_params_

Tunned_rfModel_Age = RandomForestRegressor(n_estimators = 2000,
                                           min_samples_split=5,
                                           min_samples_leaf=4,
                                           max_features="sqrt",
                                           max_depth=10,
                                           bootstrap=False,
                                           random_state=1234)

Tunned_age_accuracies = cross_val_score(estimator=Tunned_rfModel_Age,
                                        X=titanic_WithAge.loc[:, independent_variables],
                                        y=titanic_WithAge.loc[:, 'Age'],
                                        cv=10,
                                        n_jobs=2)

print("The new MEAN CV score is", round(Tunned_age_accuracies.mean(), ndigits=2))
print("The new standard deviation is", round(Tunned_age_accuracies.std(), ndigits=2))

# The MEAN CV score is 0.45
# The standard deviation is 0.06
# The model has increase by 8% and the standard deviation has decreased by 3%

# //--  Fit the tunned model in the dataset  \\-- #

Tunned_rfModel_Age.fit(titanic_WithAge.loc[:, independent_variables], titanic_WithAge.loc[:, 'Age'])

titanic_WithoutAge.loc[:, 'Age'] = Tunned_rfModel_Age.predict(X = titanic_WithoutAge.loc[:, independent_variables]).astype(int)

titanic = titanic_WithAge.append(titanic_WithoutAge).sort_values(by=['PassengerId']).reset_index(drop=True)

# We now have all observations of Age without missing values by random forest prediction

titanic.Age.describe()
plt.hist(titanic.Age)
plt.hist(titanic.Age, range=(0, 30))
plt.hist(titanic.Age, range=(30, 80))
# the distribution of the age looks almost like a normal distribution

# ==================== SEPARATE THE DATA AGAIN AND GET BACK OUR TRAIN/TEST DATASETS ==================== #

# I separate the titanic dataframe to their original train/test set
Clean_train = titanic.loc[titanic.Train_set == 1, :].reset_index(drop=True).drop("Train_set", axis=1)
Clean_test = titanic.loc[titanic.Train_set == 0, :].reset_index(drop=True).drop(["Train_set"], axis=1)

# I export them as csv file
Clean_train.to_csv("titanic_data/Clean_train.csv", index=False)
Clean_test.to_csv("titanic_data/Clean_test.csv", index=False)