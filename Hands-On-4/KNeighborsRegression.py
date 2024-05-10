import numpy as np
from math import sqrt

from KNeighborsClassifier import KNeighborsClassifier

class KNeighborsRegression(KNeighborsClassifier):
    def __init__(self, X: np.array, y: np.array, n_neighbors: int = 5):
        KNeighborsClassifier.__init__(self, X, y, n_neighbors)
        
    def predict(self, X_test: np.array) -> list:
        """Predict the class labels for the provided data.
        Returns class labels for each data sample.
        """
        neigh_ind = self.kneighbors(X_test, False)
        self.y_predict = []
        
        for nearest_neigh in neigh_ind:
            average_neigh = sum([self.y[idx] for idx in nearest_neigh]) / self.n_neighbors
            self.y_predict.append(average_neigh)
        
        return self.y_predict
    
    def score(self, y:np.array):
        mse = sum([(real - predict) ** 2 for real, predict in zip(y, self.y_predict)])/len(y)
        rmse = sqrt(mse)
        
        return mse, rmse
    