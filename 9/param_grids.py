import numpy as np

grids = {
    'xgboost': {
        'max_depth': [int(x) for x in np.linspace(2, 8, num=5)],
        'eta': [x for x in np.linspace(0.1, 1, num=5)],
        'max_leaves': [int(x) for x in np.linspace(1, 10, num=3)],
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=150, num=10)],
    },
    'knn': {
        'n_neighbors': [int(x) for x in np.linspace(1, 15, 10)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [int(x) for x in np.linspace(15, 70, 20)],
        'p': [1, 2, 3],
        'n_jobs': [-1]
    },
    'log_reg': {
        'max_iter': [20, 25, 30],
        'solver': ['liblinear', 'lbfgs', 'sag', 'saga'],
        'class_weight': ['balanced'],
        'intercept_scaling': [int(x) for x in np.linspace(1, 15, 10)],
    },
    'dec_tree': {
        'criterion': ['log_loss', 'entropy'],
        'max_depth': [int(x) for x in np.linspace(2, 15, 8)],
        'min_samples_leaf': [2, 3, 4],
        'max_features': ['sqrt', 'log2'],
    },
    'xgboost_best': {'eta': 0.55,
                     'max_depth': 3,
                     'max_leaves': 1,
                     'n_estimators': 10},
    'dec_tree_best': {'criterion': 'entropy',
                      'max_depth': 7,
                      'max_features': 'log2',
                      'min_samples_leaf': 3},
    'log_reg_best': {'intercept_scaling': 10,
                     'max_iter': 50,
                     'solver': 'liblinear',
                     'class_weight': 'balanced'}

}
