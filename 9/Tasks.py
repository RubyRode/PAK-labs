import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

df_main = pd.read_csv(r"/LessonsPAK/data/titanic/train.csv")
df_test = pd.read_csv(r"/LessonsPAK/data/titanic/test.csv")


# random forest, xgboost, logistic regression, KNN

def prepare_num(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Emb')
    df_pcl = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num


if __name__ == '__main__':
    # дропаем ненужные параметры
    df_prep_x = df_main.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    df_prep_x_tst = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df_prep_y = df_main['Survived']

    # категоризируем данные (разбиваем по колонкам разные значения, ставим 0 или 1)
    df_prep_x_num = prepare_num(df_prep_x)
    df_prep_x_tst = prepare_num(df_prep_x_tst)

    # заполняем пустые значения медианными
    data_train = df_prep_x_num.fillna(df_prep_x_num.median())
    data_test = df_prep_x_tst.fillna(df_prep_x_tst.median())

    # сплитим данные для трейна, валидации и теста
    params_train, params_valid, result_train, result_valid = train_test_split(data_train, df_prep_y,
                                                                              random_state=0, train_size=0.8)
    params_test = data_test

    classifier = SGDClassifier(random_state=0, tol=1e-3)
    param_grid = {
        'bootstrap': [True],
        'max_depth': [90, 100, 110],
        'max_features': [3, 4, 5],
        'min_samples_leaf': [3, 4, 5, 6, 7],
        'min_samples_split': [10, 12, 14, 16],
        'n_estimators': [100, 200, 300]
    }

    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(params_train, result_train)
    print(grid_search.best_params_)
    print(grid_search.score(params_train, result_train))

