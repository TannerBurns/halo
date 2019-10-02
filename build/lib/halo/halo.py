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
        self.basepath = os.path.realpath(os.getcwd())
        self.prepare()


    def split_training_set(self, features: list, labels: list, test_size=0.20, random_state=16):
        train, test, train_labels, test_labels = train_test_split(
            features, 
            labels, 
            test_size = test_size, 
            random_state = random_state
        )
        self.training = (train, train_labels)
        self.testing = (test, test_labels)
        return train, test, train_labels, test_labels


    def prepare(self, gamma=0.0001, C=float(100), solver='lbfgs', n_neighbors=3, n_clusters=2, random_state=1):
        self.svc = SVC(gamma=gamma, C=C)
        self.mlp = MLPClassifier(solver=solver, random_state=random_state)
        self.kneighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.gaussian = GaussianNB()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.lr = LinearRegression()


    def fit_all(self, train_features=None, train_labels=None):
        if not train_features and not train_labels:
            if len(self.training) == 2:
                train_features = self.training[0]
                train_labels = self.training[1]
            else:
                Exception("Error! No training data has been created or provided, " +
                "call split_training_set before this function to save to class.")
        models = [
            self.svc, self.mlp, self.kneighbors, self.gaussian, self.kmeans
        ]
        if len(set(train_labels)) == 2:
            models.append(self.lr)
        for m in models:
            m.fit(train_features, train_labels)


    def test_all(self, test_features=None, test_labels=None):
        if not test_features and not test_labels:
            if len(self.testing) == 2:
                test_features = self.testing[0]
                test_labels = self.testing[1]
            else:
                Exception("Error! No testing data has been created or provided, " +
                "call split_training_set before this function to save to class.")
        models = [
            self.svc, self.mlp, self.kneighbors, self.gaussian, self.kmeans
        ]
        if len(set(test_labels)) == 2:
            models.append(self.lr)
        for m in models:
            preds = m.predict(test_features)
            print(f'{m.__class__.__name__}: {accuracy_score(test_labels, preds)}\n')


    def training_to_dataframe(self, columns: list = []):
        if self.training:
            if not columns:
                columns = [f'{i}' for i in range(0, len(self.training[0][0]))]
            df = pandas.DataFrame(self.training[0], columns=columns)
            return df
        else:
            Exception("Error! No training data has been created, call split_training_set before this function.")


    def training_to_csv(self, filename: str = '', columns: list = []):
        if self.training:
            df = self.training_to_dataframe(columns=columns)
            if filename:
                df.to_csv(filename, index=False)
            else:
                filename = f'halo_training_{str(time.time()).replace(".", "_")}'
                df.to_csv(f'{os.path.join(self.basepath, filename)}.csv', index=False)
            return True
        else:
            Exception("Error! No training data has been created, call split_training_set before this function.")


    def testing_to_dataframe(self, columns: list = []):
        if self.testing:
            if not columns:
                columns = [f'{i}' for i in range(0, len(self.testing[0][0]))]
            df = pandas.DataFrame(self.testing[0], columns=columns)
            return df
        else:
            Exception("Error! No testing data has been created, call split_testing_set before this function.")


    def testing_to_csv(self, filename: str = '', columns: list = []):
        if self.testing:
            df = self.testing_to_dataframe(columns=columns)
            if filename:
                df.to_csv(filename, index=False)
            else:
                filename = f'halo_testing_{str(time.time()).replace(".", "_")}'
                df.to_csv(f'{os.path.join(self.basepath, filename)}.csv', index=False)
            return True
        else:
            Exception("Error! No testing data has been created, call split_testing_set before this function.")