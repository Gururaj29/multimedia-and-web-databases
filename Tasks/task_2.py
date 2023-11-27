from Clustering import dbscan
import util
import pandas as pd
from tqdm import tqdm
import os
import pathlib

TASK_ID = 2

def Execute(arguments, db):
    c = arguments["c"]
    execute_internal(c, db)

def execute_internal(c, db):
    fd = util.get_feature_model(TASK_ID)
    train_data = db.get_feature_descriptors(fd)
    label_train_data = db.get_label_feature_descriptors(fd)
    
    cluster = dbscan.DBScan(train_data, c, distances_df = db.get_distances_matrix_df(fd))
    number_of_clusters = cluster.getNumberOfClusters()
    c = min(c, number_of_clusters) # if c is more than the number of clusters, then we can only use number_of_clusters

    cluster_signifance_list = get_cluster_signifance(cluster, label_train_data, number_of_clusters, c)

    test_data = db.get_feature_descriptors(fd, train_data=False)
    test_labels = db.get_id_label_dict(train_data=False)

    output_data = {"Image IDs":[], "Predicted Labels":[], "True Labels":[]}
    predictions = {}
    for image_id, image_vector in tqdm(test_data.items()):
        predicted_label = predict_label(cluster, image_vector, label_train_data, cluster_signifance_list)
        predictions[image_id] = predicted_label
        output_data["Image IDs"].append(image_id)
        output_data["Predicted Labels"].append(predicted_label)
        output_data['True Labels'].append(test_labels.get(image_id))

    util.AnalysePredictions(db, predictions)

    pathlib.Path(util.Constants.TASK_2_LOCATION).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_data).set_index("Image IDs").to_csv(os.path.join(util.Constants.TASK_2_LOCATION, 'dbscan_%d.csv'%c))
    return

def get_cluster_signifance(cluster, label_train_data, no_of_clusters, c):
    '''
        For each label vector, computes c significant cluster
        Returns a list of dictionaries, each entry i contains a dictionary di which contains 
        cluster (ci) -> labelIds pairs indicating that all these labels have ci as their ith significant cluster
        labelIds are stored as sets for faster computations during predictions
    '''
    initiate_cluster_dicts = lambda x: {i:set() for i in range(1, x+1)}
    cluster_significance = [initiate_cluster_dicts(no_of_clusters) for i in range(c)]

    for label in label_train_data:
        c_significant_clusters = cluster.getCSignicantClusters(label_train_data[label])
        for i in range(len(c_significant_clusters)):
            cluster_id = c_significant_clusters[i]
            cluster_significance[i][cluster_id].add(label)

    return cluster_significance


def predict_label(cluster, image_vector, label_train_data, cluster_significance_list):
    most_likely_labels = get_most_likely_labels(cluster, image_vector, label_train_data, cluster_significance_list)
    return predict_label_from_most_likely_labels(image_vector, label_train_data, most_likely_labels)

def get_most_likely_labels(cluster, image_vector, label_train_data, cluster_significance_list):
    '''
        Iterates over cluster significance list and returns the shortest non-empty set of labels which have same order cluster significance as the query image
    '''
    c_significant_clusters = cluster.getCSignicantClusters(image_vector)
    likely_labels = set(label_train_data.keys())
    for i in range(len(c_significant_clusters)):
        next_label_labels = likely_labels & cluster_significance_list[i][c_significant_clusters[i]]
        if len(next_label_labels) == 0: break
        likely_labels = next_label_labels
    return likely_labels

def predict_label_from_most_likely_labels(image_vector, label_train_data, likely_labels):
    '''
        Uses euclidean distance to find the closest label vector
    '''
    return min(likely_labels, key = lambda x: util.euclidean_distance(image_vector, label_train_data[x]))