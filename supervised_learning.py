############################################################################

# OVERVIEW

# - Supervised Learning
# - Classification Example
# - Tools we are using: scikit-learn, pandas, numpy, matplotlib, tensorFlow
# - Simple Neural Network example with Tensorflow
# - Kaggle Dataset: Starbucks Locations Worldwide https://goo.gl/5GfFcU

############################################################################

# STEPS

# - Define a problem
# - Analyize Data
#   - Look at all columns
#   - Look at important features
#   - Look at the size/count
#   - Look at possible outputs
# - Prepare Data
# - Evaluate Algorithm
# - Improve Results

############################################################################

# Import libraries

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import tensorflow as tf
# from sklearn.neighbors import KNeighborsClassifier
from termcolor import colored

# Load data

file = 'dataset/starbucks_locations_worldwide.csv'
starbucks = pd.read_csv(file)

# Print data

#print(starbucks.head)
#print(starbucks.columns)
#print(starbucks.shape)

############################################################################

# Instructions

def instructions():
    print(colored('This dataset includes a record for every Starbucks or subsidiary store location currently in operation as of', 'magenta'), colored('February 2017.\n', 'red'))
    print(colored('Data Shape:', 'yellow'), colored(starbucks.shape, 'cyan'))
    print(colored('Total Countries:', 'yellow'), colored(len(starbucks.Country.unique()), 'cyan'))
    print(colored('\nTop 5 Countries:', 'yellow'))
    print(colored(starbucks.Country.value_counts().head(5), 'cyan'))
    print(colored('\nOwership:', 'yellow'))
    print(colored(starbucks['Ownership Type'].value_counts(), 'cyan'))
    print('\nPredict ownership type probability from top five countries.')

instructions()

############################################################################

# Predicting ownership type probability from top five countries.

############################################################################

# Use decision tree or knn algorithm.

############################################################################

# Visualize the top three countries on map.

############################################################################

# print(starbucks[['Country', 'Longitude', 'Latitude']])

############################################################################
