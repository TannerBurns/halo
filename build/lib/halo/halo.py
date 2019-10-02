import os
import time
import pickle
import pandas
import pandas_profiling

# sklearn import for different classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# sklearn utils
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


    def save(self, model, filename=''):
        if not filename:
            filename = os.path.join(self.basepath, f'halo_model_{str(time.time()).replace(".", "_")}.mdl')
        with open(filename, 'wb') as fout:
            pickle.dump(model, fout)
        

    def fit(self, model, train_features=None, train_labels=None):
        if not train_features and not train_labels:
            Exception("Error! No training data has been created or provided, " +
            "call split_training_set before this function to save to class.")
        model.fit(train_features, train_labels)


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


    def test(self, model, test_features=None, test_labels=None):
        if not test_features and not test_labels:
            Exception("Error! No testing data has been created or provided, " +
            "call split_training_set before this function to save to class.")
        preds = model.predict(test_features)
        print(f'{model.__class__.__name__}: {accuracy_score(test_labels, preds)}\n')


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
        if not self.training:
            Exception("Error! No training data has been created, call split_training_set before this function.")
        if not columns:
            columns = [f'{i}' for i in range(0, len(self.training[0][0]))]
        df = pandas.DataFrame(self.training[0], columns=columns)
        return df         


    def training_to_csv(self, filename: str = '', columns: list = []):
        if not self.training:
            Exception("Error! No training data has been created, call split_training_set before this function.")
        df = self.training_to_dataframe(columns=columns)
        if not filename:
            filename = os.path.join(self.basepath, f'halo_training_{str(time.time()).replace(".", "_")}.csv')
        df.to_csv(filename, index=False)
        

    def testing_to_dataframe(self, columns: list = []):
        if not self.testing:
            Exception("Error! No testing data has been created, call split_testing_set before this function.")
        if not columns:
            columns = [f'{i}' for i in range(0, len(self.testing[0][0]))]
        df = pandas.DataFrame(self.testing[0], columns=columns)
        return df
            

    def testing_to_csv(self, filename: str = '', columns: list = []):
        if not self.testing:
            Exception("Error! No testing data has been created, call split_testing_set before this function.")
        df = self.testing_to_dataframe(columns=columns)
        if not filename:
            filename = os.path.join(self.basepath, f'halo_testing_{str(time.time()).replace(".", "_")}.csv')
        df.to_csv(filename, index=False)
            
    
    def training_to_visual(self, columns: list = []):
        if not self.training:
            Exception("Error! No training data has been created, call split_training_set before this function.")
        df = self.training_to_dataframe(columns=columns)
        return df.profile_report()
    

    def testing_to_visual(self, columns: list = []):
        if not self.testing:
            Exception("Error! No testing data has been created, call split_training_set before this function.")
        df = self.testing_to_dataframe(columns=columns)
        return df.profile_report()
        