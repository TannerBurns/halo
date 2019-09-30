import os
import time
import pandas


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class halo:
    def __init__(self):
        self.prepare()

    def split_training_set(self, features, labels):
        return train_test_split(features, labels, test_size=0.20, random_state=16)

    def prepare(self, gamma=0.0001, C=float(100), solver='lbfgs', n_neighbors=3, n_clusters=2, random_state=1):
        self.basepath = os.path.realpath(os.getcwd())
        svc = SVC(gamma=gamma, C=C)
        self.svc = {
            "name": "SVC",
            "call": svc,
            "check": None,
            "skip": False
        }
        mlp = MLPClassifier(solver=solver, random_state=random_state)
        self.mlp = {
            "name": "NeuralNetMLP",
            "call": mlp,
            "check": None,
            "skip": False
        }
        kneighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.kneighbors = {
            "name": "KNearestNeighbors",
            "call": kneighbors,
            "check": None,
            "skip": False
        }
        gaussian = GaussianNB()
        self.gaussian = {
            "name": "GaussianNB",
            "call": gaussian,
            "check": None,
            "skip": False
        }
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.kmeans = {
            "name": "KMeans",
            "call": kmeans,
            "check": None,
            "skip": False
        }
        lr = LinearRegression()
        self.linearregression = {
            "name": "LinearRegression",
            "call": lr,
            "check": {
                "unique_labels": 2
            },
            "skip": False
        }
        self.models = [
            self.svc,
            self.mlp,
            self.kneighbors,
            self.gaussian,
            self.kmeans,
            self.linearregression
        ]

    def fit(self, train_features, train_labels):
        self.training = (train_features, train_labels)
        for m in self.models:
            if m["check"]:
                if "unique_labels" in m["check"]:
                    if len(set(train_labels)) == m["check"]["unique_labels"]:
                        m["call"].fit(train_features, train_labels)
                    else:
                        m["skip"] = True
            else:
                m["call"].fit(train_features, train_labels)

    def test(self, test_features, test_labels):
        for m in self.models:
            if not m["skip"]:
                preds = m["call"].predict(test_features)
                print(f'{m["name"]}: {accuracy_score(test_labels, preds)}\n')

    def to_dataframe(self, columns: list = []):
        if self.training:
            if not columns:
                columns = [f'{i}' for i in range(0, len(self.training[0][0]))]
            df = pandas.DataFrame(self.training[0], columns=columns)
            return df
        else:
            Exception("Error! No models have been fit with training data")
    
    def to_csv(self, filename: str = '', columns: list = []):
        if self.training:
            df = self.to_dataframe(columns=columns)
            if filename:
                df.to_csv(filename, index=False)
            else:
                filename = f'halo_features_{str(time.time()).replace(".", "_")}'
                df.to_csv(f'{os.path.join(self.basepath, filename)}.csv', index=False)
            return True
        else:
            Exception("Error! No models have been fit with training data")


