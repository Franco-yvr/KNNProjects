from sklearn.neighbors import NearestNeighbors


# predict the category for every test image
def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
     # create KNeighborsClassifier object
     neigh = NearestNeighbors(n_neighbors=1)

     # fit object with training data
     neigh.fit(train_image_feats)

     # predict the closest neighbour for each test element
     neighbourPred = neigh.kneighbors(test_image_feats, return_distance=False)

     # parse the predicitions
     predicted_labels = []
     for pred in neighbourPred:
         predicted_labels.append(train_labels[pred[0]])

     return predicted_labels
