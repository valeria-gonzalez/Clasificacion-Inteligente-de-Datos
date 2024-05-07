from KNeighborsClassifier import KNeighborsClassifier
import pandas as pd
import numpy as np
from math import ceil

class DataHandler:
    def __init__(self, filename: str, feature_names : list[str], class_name: str):
        self.filename = filename
        self.size = None
        self.feature_names = feature_names
        self.class_name = class_name
        self.X = None
        self.Y = None
        self.X_test = None
        self.X_train = None
        self.y_train = None
        self.y_test = None
    
    def process_data(self):
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
        
    def KNeighborsClasify(self, n_neighbors: int):
        knn = KNeighborsClassifier(self.X_train, self.y_train, n_neighbors)
        knn.fit()
        dist, idx = knn.kneighbors(self.X_test)
        results = knn.predict(self.X_test)
        
        print(f"\nDistances of nearest neighbors by test instance: \n\n {dist}\n\n")
        print(f"Indexes of nearest neighbors by test instance: \n\n {idx}\n\n")
        print(f"Class label obtained by test instance: \n\n {results}\n\n")

        
        
        