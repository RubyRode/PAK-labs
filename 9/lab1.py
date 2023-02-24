import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from param_grids import grids

data_labels = pd.read_csv('/home/ruby/PycharmProjects/PAK-labs/titanic/titanic_prepared.csv')
data = data_labels.drop('Unnamed: 0', axis=1)
labels = data['label']
data = data.drop('label', axis=1)

train_x, test_x, train_y, test_y = train_test_split(data, labels, random_state=0, train_size=0.9)

# Logistic Regression
logistic_regression_model = LogisticRegression(**grids['log_reg_best'])
logistic_regression_model.fit(test_x, test_y)
print('Logistic Regression accuracy: ', logistic_regression_model.score(test_x, test_y))

# Decision Tree
decision_tree_model = DecisionTreeClassifier(**grids['dec_tree_best'])
decision_tree_model.fit(test_x, test_y)
print('DecisionTree accuracy: ', decision_tree_model.score(test_x, test_y))

# XGBoost
xgboost_model = xgboost.XGBClassifier(**grids['xgboost_best'])
xgboost_model.fit(test_x, test_y)
print('XGBoost accuracy: ', xgboost_model.score(test_x, test_y))

# Logistic Regression 2 parameters
num = 2
importances = decision_tree_model.feature_importances_
indices = np.argsort(importances)
features = train_x.columns[indices[(len(indices) - 1 - num)]: len(indices) - 1]
indices = indices[(len(indices) - 2 - num): len(indices) - 1]
decision_tree_model.fit(train_x[features], train_y)
print(features)
print(indices)
print(train_x.columns)
predict = decision_tree_model.predict(train_x[features])
print("accuracy: ", decision_tree_model.score(train_y, predict))