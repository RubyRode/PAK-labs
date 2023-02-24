from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_test = pd.read_csv('/home/ruby/PycharmProjects/PAK-labs/titanic/wells_info.csv')

data_test = data_test.drop(['PermitDate', 'SpudDate', 'CompletionDate', 'FirstProductionDate', 'operatorNameIHS',
                            'formation', 'BasinName', 'StateName', 'CountyName'], axis=1)
data_test = data_test.to_numpy()

def cluster(m, data_t):
    clusters = m.fit_predict(data_test)
    plt.figure(figsize=(6, 6))
    for cl in np.unique(clusters):
        data_ = data_t[clusters == cl]
        plt.scatter(data_[:, 0], data_[:, 1])
    plt.legend(loc=2)
    plt.show()


model = KMeans(n_clusters=3)
cluster(model, data_test)

model = DBSCAN(eps=0.08, min_samples=1)
cluster(model, data_test)

model = AgglomerativeClustering(n_clusters=3)
cluster(model, data_test)

model = SpectralClustering(n_jobs=-1)
cluster(model, data_test)

