from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

df_main = pd.read_csv(r"/home/ruby/PycharmProjects/PAK-labs/titanic/train.csv")

def prepare_num(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Emb')
    df_pcl = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num


if __name__ == "__main__":
    df_prep_x = df_main.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    df_prep_y = df_main['Survived']
    # категоризируем данные (разбиваем по колонкам разные значения, ставим 0 или 1)
    df_prep_x_num = prepare_num(df_prep_x)

    # заполняем пустые значения медианными
    data_train = df_prep_x_num.fillna(df_prep_x_num.median())
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    # сплитим данные для трейна, валидации и теста
    train_x, valid_x, train_y, valid_y = train_test_split(data_train, df_prep_y,
                                                          random_state=0, train_size=0.6)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, random_state=0, train_size=0.8)

    # создаем модель
    param_grid = {
        'max_iter': [int(x) for x in np.linspace(100, 1000, 100)],
        'solver': ['liblinear'],
        'class_weight': ['balanced'],
        'penalty': ['l2', 'l1'],
        }

    model = LogisticRegression()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(valid_x, valid_y)
    print(grid_search.best_params_)
    print("grid_search best score: ", grid_search.best_score_)

    pr = grid_search.best_params_
    model = LogisticRegression(**pr)
    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    print(f'Accuracy: {accuracy_score(test_y, predict)}')
