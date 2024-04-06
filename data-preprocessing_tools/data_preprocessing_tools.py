#!/usr/bin/python3

# Importing the Libraries
import numpy as np                    # A library for manipulating multidimensional arrays and mathematical calculations in Python 
import matplotlib.pyplot as plt       # 2D plotting library for high-quality graphics 
import pandas as pd                   # A library offering easy-to-use, high-performance data structures and data manipulation tools

# Importing the dataset
dataset = pd.read_csv('Data.csv')     # Reads data from a CSV file and loads it into a DataFrame
X = dataset.iloc[:, :-1].values       # Characteristics matrix contains the characteristics or explanatory variables used to predict the target variable
y = dataset.iloc[:, -1].values        # Dependent variable vector is a target variable you are trying to predict using the characteristics matrix - last column

print(X)
print(y)

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum()

# Print the number of missing entries in each column
print("Missing data: \n", missing_data)

# Taking care of missing data
"""The SimpleImputer class is used to manage missing values in data by replacing them with a specified value."""
from sklearn.impute import SimpleImputer

"""
Missing values are represented by np.nan, which is NumPy's convention for missing values.
Strategy='mean' specifies that the strategy for replacing missing values is to replace them
with the average of the values present in the respective column.
"""
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

"""Calculates the average of the values in columns 1 to 2 of the matrix X."""
imputer.fit(X[:, 1:3])

"""Replaces the missing values in columns 1 to 2 of the X matrix with the averages calculated previously."""
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

# Encoding categorical data

## Encoding the Independent Variable
"""Apply different transformers to different columns in the dataset."""
from sklearn.compose import ColumnTransformer

"""Encode categorical variables as binary variables."""
from sklearn.preprocessing import OneHotEncoder

"""
Process the categorical variables in the first column and transform them into binary variables.
The other columns will simply be passed on unchanged using remainder='passthrough'.
"""
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

"""Applies the transformation to X and converts the result into a NumPy array and assigns it to variable X."""
X = np.array(ct.fit_transform(X))

print(X)

## Encoding the Dependent Variable
"""Encode categorical values into numerical values."""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

# Splitting the dataset into the Training set and Test set
"""
Dividing data into training and test groups is super important.
It helps to ensure that your model works well with new data and to estimate its performance fairly.
It also helps to spot problems such as overlearning and to fine-tune the model parameters.
"""

"""Used to divide data into training and test sets."""
from sklearn.model_selection import train_test_split

"""
X_train: the set of training characteristics.
X_test: the set of test characteristics.
y_train: the target values corresponding to the training set.
y_test: the target values corresponding to the test set.

test_size = 0.2 specifies that 20% of the data will be used as the test set,
while 80% will be used as the training set

random_state = 1 means that if you run the code several times with the same random seed,
you will always obtain the same division of the data.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

# Feature Scaling
"""
A data pre-processing technique widely used in machine learning to normalise the features of a dataset within a specific range.
This typically involves adjusting feature values to fall within a given range or have a specific distribution.
The two most common methods of feature are normalisation (or min-max scaling) and standardisation.
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:, 3:] = scaler.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scaler.transform(X_test[:, 3:])

print(X_train)

print(X_test)