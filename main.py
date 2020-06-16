import pandas as pd
import numpy as np
import scipy
import sklearn
import csv
from sklearn.linear_model import RidgeCV, Lasso, LassoCV, ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt

def calc_function(values, function):
    switch = {
        '1': values,
        'quad': np.square(values),
        'exp': np.exp(values),
        'cos': np.cos(values),
    }
    return switch.get(function, 0)

dataframe = pd.read_csv('train.csv')
y = dataframe.iloc[:, 1]
x = dataframe.iloc[:, 2:]
x = np.array(x)
y = np.array(y)

transforms = np.array(['1' , 'quad' , 'exp' , 'cos'])
x_transformed = np.zeros((x.shape[0],21))
x_transformed[:,20] = np.ones((x_transformed.shape[0]))

for i, transform in enumerate(transforms):
    for j in range(0,5):
        x_transformed[:,i*5+j] = calc_function(x[:,j], transform)

lambdas = list(np.linspace(20,25,100))
lambdasLasso = list(np.linspace(1e-10,1e3,100))#[0.05555555555555556]

clf2 = LassoCV(alphas=[0.05555555555555556],cv=10, fit_intercept=False, max_iter=500000).fit(x_transformed, y)

lambda_used = clf2.alpha_
weights = clf2.coef_

with open('./submission.csv', 'w', newline='') as csvfile:
    submission_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for weight in weights:
        submission_file.writerow([weight])