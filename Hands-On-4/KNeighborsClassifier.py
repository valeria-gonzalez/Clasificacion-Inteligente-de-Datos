from Maths import Maths 
import numpy as np
import heapq
from collections import defaultdict

class KNeighborsClassifier:
    def __init__(self, X: np.array, y: np.array, n_neighbors: int = 5):
        """Returns the fitted k-nearest neighbors classifier."""
        self.X = X # training data [n_samples][n_features]
        self.y = y # target values [n_samples]
        self.n_neighbors = n_neighbors
        self.classes = None # class labels known to the classifier
        self.n_features_in = None # no of features seen during fit
        self.n_samples_fit = None # number of samples in fitted data
        
    def fit(self) -> None:
        """Fit the k-nearest neighbors classifier from the training set.
        Returns self the fitted k nearest neighbors classifier.
        """
        self.n_features_in = len(self.X[0])
        self.classes = np.unique(self.y)
    
    def kneighbors(self, X_test: np.array, return_distance: bool = True) -> list:
        """Find the k-neighbors of multiple point queries.
        Returns two different arrays with indices of and distances to the neighbors of each point.
        """
        neigh_dist = []
        neigh_ind = []
        
        for query in X_test:
            if return_distance:
                nearest_dist, nearest_ind = self.__calcBruteDistance(query)
            else:
                nearest_ind = self.__calcBruteDistance(query, return_distance)

            neigh_ind.append(nearest_ind)
            if return_distance:
                neigh_dist.append(nearest_dist) 
            
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
    
    def predict(self, X_test: np.array) -> list:
        """Predict the class labels for the provided data.
        Returns class labels for each data sample.
        """
        neigh_ind = self.kneighbors(X_test, False)
        y_labels = []
        
        for nearest_neigh in neigh_ind:
            class_count = defaultdict(int)
            for index in nearest_neigh:
                class_count[self.y[index]] += 1
            class_label = max(class_count, key = class_count.get)
            y_labels.append(class_label)
        
        return y_labels
            
    def score(self) -> float:
        """Return the mean accuracy on the given test data and labels
        Return mean accuracy of self.predict w.r.t y
        """
        pass
    
    def __calcBruteDistance(self, X_test: np.array, return_distance: bool = True):
        """Find the k-neighbors of a point.
        Returns two different arrays with indices of and distances to the neighbors of each point.
        """
        euclidean_distances = []
        maths = Maths()
        
        for index, instance in enumerate(self.X):
            distance = maths.euclideanDistance(instance, X_test)
            heapq.heappush(euclidean_distances, (distance, index))
        
        neigh_dist = []
        neigh_ind = []
        
        for _ in range(self.n_neighbors):
            distance, index = heapq.heappop(euclidean_distances)
            neigh_ind.append(index)
            if return_distance:
                neigh_dist.append(distance)
            
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
            
        
            
        
        