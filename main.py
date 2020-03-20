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
lambdasLasso = list(np.linspace(0,0.1,10))

clf = RidgeCV(alphas=lambdas, cv=10, fit_intercept=False).fit(x_transformed, y)
clf2 = LassoCV(alphas=[0.05555555555555556],cv=10, fit_intercept=False).fit(x_transformed, y)
clf3 = ElasticNetCV(alphas=[0.031578947368421054], l1_ratio=0, cv=10, fit_intercept=False).fit(x_transformed, y)
lambda_used = clf.alpha_
weights = clf2.coef_




with open('./submission.csv', 'w', newline='') as csvfile:
    submission_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for weight in weights:
        submission_file.writerow([weight])