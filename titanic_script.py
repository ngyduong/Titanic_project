# ==================== PACKAGES ==================== #

import pandas as pd
# import os
import numpy as np
import sklearn

# ==================== IMPORT DATA ==================== #

test = pd.read_csv("titanic_data/test.csv")
train = pd.read_csv("titanic_data/train.csv")

# ==================== CLEANING DATA ==================== #

#Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
train['Deck']=train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))









