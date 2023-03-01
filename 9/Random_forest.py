import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

df_main = pd.read_csv(r"/home/ruby/PycharmProjects/PAK-labs/titanic/train.csv")


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
    df_prep_y = df_main['Survived']
    # категоризируем данные (разбиваем по колонкам разные значения, ставим 0 или 1)
    df_prep_x_num = prepare_num(df_prep_x)

    # заполняем пустые значения медианными
    data_train = df_prep_x_num.fillna(df_prep_x_num.median())

    # сплитим данные для трейна, валидации и теста
    train_x, valid_x, train_y, valid_y = train_test_split(data_train, df_prep_y,
                                                          random_state=0, train_size=0.6)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, random_state=0, train_size=0.8)

    # задаем сетку для побора гиперпараметров
    param_grid = {
        'bootstrap': [True],
        'max_depth': [int(x) for x in np.linspace(50, 180, num=15)],
        'max_features': [3, 5, 7],
        'min_samples_leaf': [int(x) for x in np.linspace(start=5, stop=14, num=3)],
        'min_samples_split': [int(x) for x in np.linspace(start=7, stop=18, num=4)],
        'n_estimators': [int(x) for x in np.linspace(start=1, stop=100, num=10)],
        'criterion': ['log_loss']
    }

    # подбираем гиперапараметры
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(valid_x, valid_y)

    print(grid_search.best_params_, grid_search.best_score_)

    # берем лучшие параметры
    pr = grid_search.best_params_
    model = RandomForestClassifier(criterion=pr['criterion'],
                                   max_depth=pr['max_depth'],
                                   min_samples_split=pr['min_samples_split'],
                                   n_estimators=pr['n_estimators'],
                                   max_features=pr['max_features'],
                                   min_samples_leaf=pr['min_samples_leaf'],
                                   bootstrap=True)
    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    # получаем точность
    print("accuracy: ", accuracy_score(test_y, predict))

    # 2 задание
    num = 8
    importances = model.feature_importances_
    indices = np.argsort(importances)
    features = data_train.columns[indices[(len(indices) - 1 - num)]: len(indices) - 1]
    new_test_data = data_train[features]
    new_test_data_y = df_prep_y
    model.fit(new_test_data, new_test_data_y)
    predict = model.predict(new_test_data)
    print("accuracy: ", accuracy_score(new_test_data_y, predict))
