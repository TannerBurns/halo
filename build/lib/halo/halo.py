import os
import pickle
import asyncio
import pandas
import pandas_profiling
# utils
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Tuple
# sklearn utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# sklearn classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN



class Halo(object):
    '''Halo -- Base class'''

    def __init__(self, workers: int= 32):
        self.__version__ = '0.34.7'
        self.basepath = os.path.realpath(os.getcwd())
        self._num_workers = workers
    
    async def _bulk(self, fn: Callable, args: list):
        '''_bulk -- run function in threadpoolexecutor
        
           fn   -- {Callable} function to run in bulk
           args -- {list} arguments to be distributed to function

           return -- list of results from function
        '''
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = [self.loop.run_in_executor(executor, partial(fn, a)) for a in args if a]
            await asyncio.gather(*futures)
            return [f.result() for f in futures]
    
    def multitask(self, fn:Callable, args: list):
        '''multitask -- run function in bulk
        
           fn   -- {Callable} function to run in bulk
           args -- {list} arguments to be distributed to function

           return -- list of results from function
        '''
        self.loop = asyncio.new_event_loop()
        return self.loop.run_until_complete(self._bulk(fn ,args))
    
    
class BaseParser(Halo):
    '''BaseParser -- Base class for Parsing Tools'''

    def __init__(self, parser_fn: Callable, parser_labels: list= list(), *args: list, **kwargs: dict):
        # required
        super().__init__(*args, **kwargs)
        self.parse = parser_fn
        if not self.parse:
            raise Exception('ERROR: Parsing function is required upon initialization')

        # optional
        self.parser_labels = parser_labels

    def parse(self, arg):
        '''parse -- parse an object, returning a feature vector --- this function must be provided on initialization
        
           arg -- argument to use in parsing function
        '''
        return list()
    
    def multiparse(self, args: list):
        return self.multitask(self.parse, args)


class Sentinel(BaseParser):
    def __init__(self, parser_fn: Callable, parser_labels: list= list()):
        super().__init__(parser_fn, parser_labels=parser_labels)


class BaseCAM(Halo):
    '''BaseCAM -- Base class for Clustering and Modeling Tools'''

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
    
    def save(self, model, filename=''):
        if not filename:
            filename = os.path.join(
                self.basepath, 
                f'{model.__class__.__name__}_{str(uuid4()).replace("-", "_")}.mdl'
            )
        with open(filename, 'wb') as fout:
            pickle.dump(model, fout)
           
    def load(self, filepath):
        with open(filepath, 'rb') as fin:
            return pickle.load(fin)
    
    def split_feature_set(self, features: list, labels: list, test_size=0.20, random_state=16):
        train, test, train_labels, test_labels = train_test_split(
            features, 
            labels, 
            test_size = test_size, 
            random_state = random_state
        )
        self.training = (train, train_labels)
        self.testing = (test, test_labels)
        return train, test, train_labels, test_labels
    
    def fit(self, model, train_features=None, train_labels=None):
        if not train_features and not train_labels:
            if len(self.training) == 2:
                train_features = self.training[0]
                train_labels = self.training[1]
            else:
                Exception("Error! No training data has been created or provided, " +
                "call split_training_set before this function to save to class.")
        model.fit(train_features, train_labels)

    def test(self, model, test_features=None, test_labels=None):
        if not test_features and not test_labels:
            if len(self.testing) == 2:
                test_features = self.testing[0]
                test_labels = self.testing[1]
            else:
                Exception("Error! No testing data has been created or provided, " +
                "call split_training_set before this function to save to class.")
        preds = model.predict(test_features)
        print(f'{model.__class__.__name__}: {accuracy_score(test_labels, preds)}\n')


class Covenant(BaseCAM):

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.prepare()
    
    def prepare(self, gamma=0.0001, C=float(100), solver='lbfgs', n_neighbors=3, n_clusters=2, random_state=1):
        self.svc = SVC(gamma=gamma, C=C)
        self.mlp = MLPClassifier(solver=solver, random_state=random_state)
        self.kneighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.gaussian = GaussianNB()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def _fit_in_bulk(self, args: Tuple[Callable, list, list]):
        '''_fit_in_bulk -- helper for fit_all

            args -- {list} (const)
                index - value
                0 - model
                1 - train_features
                2 - train_labels
            
        '''
        args[0].fit(args[1], args[2])

    def fit_all(self, train_features=None, train_labels=None):
        if not train_features and not train_labels:
            if len(self.training) == 2:
                train_features = self.training[0]
                train_labels = self.training[1]
            else:
                Exception("Error! No training data has been created or provided, " +
                "call split_training_set before this function to save to class.")
        args = [
            (self.svc, train_features, train_labels),
            (self.mlp, train_features, train_labels),
            (self.kneighbors, train_features, train_labels),
            (self.gaussian, train_features, train_labels),
            (self.kmeans, train_features, train_labels)
        ]
        self.multitask(self._fit_in_bulk, args)
    
    def _test_in_bulk(self, args: Tuple[Callable, list, list]):
        '''_test_in_bulk -- helper for test_all

            args -- {list} (const)
                index - value
                0 - model
                1 - test_features
                2 - test_labels
            
        '''
        preds = args[0].predict(args[1])
        print(f'{m.__class__.__name__}: {accuracy_score(args[2], preds)}\n')

    def test_all(self, test_features=None, test_labels=None):
        if not test_features and not test_labels:
            if len(self.testing) == 2:
                test_features = self.testing[0]
                test_labels = self.testing[1]
            else:
                Exception("Error! No testing data has been created or provided, " +
                "call split_training_set before this function to save to class.")
        args = [
            (self.svc, test_features, test_labels),
            (self.mlp, test_features, test_labels),
            (self.kneighbors, test_features, test_labels),
            (self.gaussian, test_features, test_labels),
            (self.kmeans, test_features, test_labels)
        ]
        self.multitask(self._test_in_bulk, args)
            
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
            filename = os.path.join(self.basepath, f'halo_trainingset_{str(time.time()).replace(".", "_")}.csv')
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
            filename = os.path.join(self.basepath, f'halo_testingset_{str(time.time()).replace(".", "_")}.csv')
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


class Flood(BaseCAM):

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.prepare()
    
    def prepare(self, n_clusters=2, epcs=3, min_samples=2, random_state=1):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.dbscan = DBSCAN(eps=epcs, min_samples=min_samples)
    
    def fit_and_map(self, cluster, train_features, train_names):
        if len(train_features) == len(train_names):
            cluster.fit(train_features)
            return {train_names[i]:{"label": float(cluster.labels_[i])} for i in range(0, len(train_names))}
        else:
            Exception("Error! Cannot fit and map the cluster, len of features != len of names")
    
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