# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== DATA MANIPULATION ==================== #

train = pd.read_csv("titanic_data/clean_data/Clean_train.csv")
train = train.drop(['PassengerId', 'Name', 'Age', 'female', 'male', 'Pclass_1',
                    'Pclass_2', 'Pclass_3', 'Dr', 'Master', 'Miss', 'Mr', 'Mrs','Nobility',
                    'Officer', 'big_family', 'small_family', 'solo', 'Deck', 'Age_Randomforest',
                    'Age_SVM', "C", "Q", "S", "Fare"], axis=1)

# ==================== DATA PROCESSING ==================== #

# //-- Pclass \\-- #

def Pclass_process(x):
    if x == 1:
        return('1st')
    elif x == 2:
        return('2nd')
    else:
        return('3rd')

train.loc[:, "Pclass"] = train.Pclass.apply(lambda x: Pclass_process(x))

# //-- Survived \\-- #

def Survived_process(x):
    if x == 1:
        return("Survived")
    else:
        return("Died")

train.loc[:, "Survived"] = train.Survived.apply(lambda x: Survived_process(x))

# ==================== Statistical analysis ==================== #

# //-- Survival vs Pclass \\-- #

pd.crosstab(train['Pclass'],train['Survived']).plot.bar()
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Cross analysis between Survived an Pclass')
plt.legend(["Has died", "Has survived"])

# //-- Survival vs Sex \\-- #

pd.crosstab(train['Sex'],train['Survived']).plot.bar()
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.title('Cross analysis between Survived an Sex')
plt.legend(["Has died", "Has survived"])

# //-- Pclass vs Sex \\-- #

pd.crosstab(train['Pclass'],train['Sex']).plot.bar()
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Cross analysis between Pclass an Sex')

