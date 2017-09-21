# OVERVIEW
# 1. Supervised Learning
# 2. Classification Example
# 3. Tools we are using: scikit-learn, pandas, numpy, matplotlib
# 4. Decision Tree or KNN algorithm
# 5. Kaggle Dataset: Starbucks Locations Worldwide https://goo.gl/5GfFcU

# STEPS
# 1. Define a problem
# 2. Analyize Data
    # a. Look at all columns
    # b. Look at important features
    # c. Look at the size/count
    # d. Look at possible outputs
# 3. Prepare Data
# 4. Evaluate Algorithm
# 5. Improve Results


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from termcolor import colored

# load data
file = 'dataset/starbucks_locations_worldwide.csv'
starbucks = pd.read_csv(file)

# instructions
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

# predicting ownership type probability from top five countries
# use decision tree or knn algorithm

# visualize the top three countries on map

# print(starbucks[['Country', 'Longitude', 'Latitude']])