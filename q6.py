print('###################### Question 6  ######################')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint


#Load dataset
df = pd.read_csv(
    'grbs.txt',
    delim_whitespace=True, skiprows=2, na_values=-1,
    names=['Type', 'Catalog', 'Redshift', 'T90', 'log (M*/Mo)', 'SFR', 'log (Z/Zo)', 'SSFR', 'AV'])

# Cleanup dataset

# drop XRFs
df.drop(df[df['Type'] != 'GRB'].index, axis=0, inplace=True)

# assign Label based on T90: if T90> 10s, label is True. i.e. for long GRBs, label is true. for short GRBs label is false
df['Label'] = df.apply(lambda row: row['T90'] > 10, axis=1)

# drop unusable columns
df.drop(['Type', 'Catalog', 'T90'], axis=1, inplace=True)

# replace NaN with 0
dfi = df.fillna(0)

# code for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def predict_probs(X, theta):
    return sigmoid(np.dot(X, theta))

def predict(X, theta, threshold=0.5):
    return predict_probs(X, theta) >= threshold

def fit(X, y, learning_rate=0.01, iterations=5000):
    theta = np.zeros(X.shape[1])

    for i in range(iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= learning_rate * gradient

    return theta

# Split dataset and drop parameters that won't be used to find the weights
X = dfi.drop(['Label', 'AV', 'SSFR',], axis=1)
y = dfi['Label']

# via custom
theta = fit(X, y)
predictions = pd.Series(predict(X, theta))

yg = y.value_counts()
pg = predictions.value_counts()

ind = np.arange(2) 
width = 0.35       
plt.bar(ind, (yg[True], yg[False]), width, label='Actual')
plt.bar(ind + width, (pg[True], pg.get(False, 0)), width, label='Predicted')

plt.ylabel('Count')
plt.title('GRB Classification')

plt.xticks(ind + width / 2, ('Long', 'Short'))
plt.legend(loc='best')
plt.savefig('grb_hist.png')

pprint(dict(zip(X.columns, theta)))
