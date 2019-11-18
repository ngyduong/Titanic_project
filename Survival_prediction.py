# ==================== PACKAGES ==================== #


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV


# ==================== IMPORT MANIPULATION ==================== #


test = pd.read_csv("titanic_data/Clean_test.csv")
train = pd.read_csv("titanic_data/Clean_train.csv")


# ==================== SURVIVAL PROCESSING ==================== #

