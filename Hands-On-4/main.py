from DataHandler import DataHandler

def main():
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
    handler.process_data()
    handler.train_test_split(train_percent)
    handler.KNeighborsClasify(n_neighbors)
    
    
    
if __name__ == "__main__":
    main()