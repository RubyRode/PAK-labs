from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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

    random_grid = {
        'n_neighbors': [int(x) for x in np.linspace(1, 15, 10)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [int(x) for x in np.linspace(15, 70, 20)],
        'p': [1, 2, 3],
        'n_jobs': [-1]
    }

    kn = KNeighborsClassifier()
    model_r = GridSearchCV(kn, random_grid, cv=3, verbose=2, n_jobs=-1, scoring='f1')
    model_r.fit(valid_x, valid_y)
    print(model_r.best_params_)
    print(f'accuracy: {model_r.best_score_}')

    pr = model_r.best_params_
    model = KNeighborsClassifier(**pr)
    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    print(accuracy_score(test_y, predict))