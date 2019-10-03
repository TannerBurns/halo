import os
import time
import pickle

# sklearn clustering imports
from sklearn.cluster import KMeans, DBSCAN


class Flood:
    def __init__(self):
        self.basepath = os.path.realpath(os.getcwd())
        self.prepare()
    
    
    def prepare(self, n_clusters=2, epcs=3, min_samples=2, random_state=1):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.dbscan = DBSCAN(eps=epcs, min_samples=min_samples)
    

    def save(self, model, filename=''):
        if not filename:
            filename = os.path.join(self.basepath, f'cluster_{str(time.time()).replace(".", "_")}.cstr')
        with open(filename, 'wb') as fout:
            pickle.dump(model, fout)
     
        
    def load(self, filepath):
        with open(filepath, 'rb') as fin:
            return pickle.load(fin)
    

    def fit_and_map(self, cluster, train_features, train_names):
        if len(train_features) == len(train_names):
            cluster.fit(train_features)
            return {train_names[i]:{"label": float(cluster.labels_[i])} for i in range(0, len(train_names))}
        else:
            Exception("Error! Cannot fit and map clustering, len of features != len of names")
    
    def fit_and_map_all(self, train_features, train_names):
        if len(train_features) == len(train_names):
            algs = [self.kmeans, self.dbscan]
            data = {}
            for a in algs:
                a.fit(train_features)
                data.update(
                    { 
                        f'{a.__class__.__name__}': {
                            train_names[i]: { 
                                "label": float(a.labels_[i])
                            } for i in range(0, len(train_names))
                        }
                    }
                )
            return data
        else:
            Exception("Error! Cannot fit and map clustering, len of features != len of names")
        

    
    
    
