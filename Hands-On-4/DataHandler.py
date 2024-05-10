from KNeighborsClassifier import KNeighborsClassifier
from KNeighborsRegression import KNeighborsRegression
import pandas as pd
import numpy as np
from math import ceil

class DataHandler:
    def __init__(self, filename: str, feature_names : list[str], class_name: str):
        self.filename = filename
        self.size = None
        self.X = None
        self.Y = None
        self.feature_names = feature_names
        self.class_name = class_name
        self.X_test = None
        self.X_train = None
        self.y_train = None
        self.y_test = None
    
    def fit(self):
        df = pd.read_csv(self.filename)
        self.X = df[self.feature_names]
        self.X = [np.array(row) for row in self.X.itertuples(index=False)]
        self.y = df[self.class_name].to_numpy()
        self.size = len(self.y)
        
    def train_test_split(self, train_percent: int)-> tuple:
        train_size = ceil(train_percent * self.size / 100)
        self.X_train = self.X[:train_size]
        self.y_train = self.y[:train_size]
        self.X_test = self.X[train_size:]
        self.y_test = self.y[train_size:]
        
    def printResults(self, neigh_dist, neigh_ind, knn_result_class, title):
        print(f"------------------------------ {title} ------------------------------")
        
        print(f"\nDescription of test instances: \n\n")
        for index, instance in enumerate(self.X_test):
            print(f"[{index}] : {instance}")
        print("\n")
        
        print(f"\nDistances of nearest neighbors by test instance: \n\n")
        for index, distances in enumerate(neigh_dist):
            print(f"[{index}] : {distances}")
        print("\n")
            
        print(f"Indexes of nearest neighbors by test instance: \n\n")
        for index, n_ind in enumerate(neigh_ind):
            print(f"[{index}] : {n_ind}")
        print("\n")
        
        print(f"Class label obtained by test instance: \n\n")
        for index, result_class in enumerate(knn_result_class):
            print(f"[{index}] : {result_class}")
        print("\n")
            
    def kNeighborsClasify(self, n_neighbors: int):
        knn = KNeighborsClassifier(self.X_train, self.y_train, n_neighbors)
        knn.fit()
        neigh_dist, neigh_ind = knn.kneighbors(self.X_test)
        knn_result_class = knn.predict(self.X_test)
        accuracy = knn.score(self.y_test)
        self.printResults(neigh_dist, neigh_ind, knn_result_class, "Classification")
        print(f"Accuracy: {accuracy}\n\n")
        
    def kNeighborsRegression(self, n_neighbors: int):
        knn = KNeighborsRegression(self.X_train, self.y_train, n_neighbors)
        knn.fit()
        neigh_dist, neigh_ind = knn.kneighbors(self.X_test)
        knn_result_class = knn.predict(self.X_test)
        mse, rmse = knn.score(self.y_test)
        self.printResults(neigh_dist, neigh_ind, knn_result_class, "Regression")
        print(f"MSE: {mse}\n\n")
        print(f"RMSE: {rmse}\n\n")