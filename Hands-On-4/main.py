from DataHandler import DataHandler

def main():
    knn_classify()
    knn_regression()
        
        
def knn_classify():
    filename = 'iris2.csv'
    feature_names = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ]
    class_name = 'species'
    train_percent = 65
    n_neighbors = 3
    handler = DataHandler(filename, feature_names, class_name)
    handler.fit()
    handler.train_test_split(train_percent)
    handler.kNeighborsClasify(n_neighbors)
    
    
def knn_regression():
    filename = 'iris3.csv'
    feature_names = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ]
    class_name = 'species'
    train_percent = 65
    n_neighbors = 3  
    handler = DataHandler(filename, feature_names, class_name)
    handler.fit()
    handler.train_test_split(train_percent)
    handler.kNeighborsRegression(n_neighbors)
    
    
if __name__ == "__main__":
    main()