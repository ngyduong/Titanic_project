# Author: NGUYEN DUONG
# Project: Titanic Machine Learning from Disaster

# ==================== PACKAGES ==================== #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==================== DATA MANIPULATION ==================== #

train = pd.read_csv("titanic_data/clean_data/Clean_train.csv")
train = train.drop(['PassengerId', 'Name', 'Age', 'female', 'male', 'Pclass_1',
                    'Pclass_2', 'Pclass_3', 'Dr', 'Master', 'Miss', 'Mr', 'Mrs','Nobility',
                    'Officer', 'big_family', 'small_family', 'solo', 'Deck', 'Age_Randomforest',
                    'Age_SVM', "C", "Q", "S"], axis=1)

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

# //-- Title vs Survived \\-- #

pd.crosstab(train['Title'],train['Survived']).plot.bar()
plt.xlabel('Title')
plt.ylabel('Frequency')
plt.title('Cross analysis between Title an Sex')

# //-- Age vs Has Survived \\-- #

pd.crosstab(train['Age_replace'], train.loc[train.Survived=="Survived", 'Survived']).plot.bar(color="green")
plt.xlabel('Age')
plt.ylabel('Frequency')

# //-- Age vs Has Died \\-- #

pd.crosstab(train['Age_replace'], train.loc[train.Survived=="Died", 'Survived']).plot.bar(color='red')
plt.xlabel('Age')
plt.ylabel('Frequency')

# //-- Famsize vs Survived \\-- #

pd.crosstab(train['Famsize'],train['Survived']).plot.bar()
plt.xlabel('Title')
plt.ylabel('Frequency')
plt.title('Cross analysis between Family size an Survived')

# //-- Boxplot Age by Survival \\-- #

sns.boxplot(x='Survived', y='Age_replace', data=train)
ax = sns.stripplot(x='Survived', y='Age_replace', data=train, color="black", jitter=0.2, size=2.5)
plt.title("Age Boxplot by Survival", loc="left")


# //-- Boxplot Fare by Survival \\-- #

sns.boxplot(x='Survived', y='Fare', data=train)
ax = sns.stripplot(x='Survived', y='Fare', data=train, color="green", jitter=0.2, size=2.5)
plt.title("Fare Boxplot by Survival", loc="left")

# //-- Boxplot Fare by Pclass \\-- #

sns.boxplot(x='Pclass', y='Fare', data=train)
ax = sns.stripplot(x='Pclass', y='Fare', data=train, color="purple", jitter=0.2, size=2.5)
plt.title("Fare Boxplot by Pclass", loc="left")

# //-- Boxplot Fare by Title \\-- #

sns.boxplot(x='Title', y='Fare', data=train)
ax = sns.stripplot(x='Title', y='Fare', data=train, color="purple", jitter=0.2, size=2.5)
plt.title("Fare Boxplot by Title", loc="left")