from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
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

    # сплитим данные для трейна, валидации и теста
    train_x, valid_x, train_y, valid_y = train_test_split(data_train, df_prep_y,
                                                          random_state=0, train_size=0.6)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, random_state=0, train_size=0.8)

    param_grid = {
        'max_depth': [int(x) for x in np.linspace(3, 8)],
        'eta': [x for x in np.linspace(0.1, 1, num=5)],
        'tree_method': ['exact', 'gpu_hist'],
        'max_leaves': [int(x) for x in np.linspace(1, 10, num=5)],
        'predictor': ['gpu_predictor'],

        'n_estimators': [int(x) for x in np.linspace(start=10, stop=150, num=10)],
    }

    xg = XGBClassifier()
    grid_search = GridSearchCV(xg, param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(valid_x, valid_y)

    print(grid_search.best_params_)
    print("grid_search best score: ", grid_search.best_score_)

    pr = grid_search.best_params_
    model = XGBClassifier(**pr)
    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    print(f"Accuracy: {accuracy_score(test_y, predict)}")
