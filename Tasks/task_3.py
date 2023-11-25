import util
from Classifiers import nn
from Classifiers import ppr
from Classifiers import decision_trees
from tqdm import tqdm
import time
import pandas as pd
import os
import pathlib

TASK_ID = 3

def Execute(arguments, db):
    if not validate_input_arguments(arguments):
        return
    classifier, query_image_ids = extract_arguments(arguments)
    execute_internal(classifier, query_image_ids, db)

def validate_input_arguments(arguments):
    if 'classifier' not in arguments or not arguments['classifier']:
        print("Error: need a type of classifier")
        return False
    classifier_type = util.classifier_cli_to_constants(arguments['classifier'])
    if classifier_type == util.Constants.NearestNeighborClassifier and not arguments.get('m'):
        print("Error: need m for NN classifier")
        return False
    if classifier_type == util.Constants.PersonalizedPageRankClassifier and not arguments.get('p'):
        print("Error: need random jump probability for PPR classifier")
        return False
    return True
    
def extract_arguments(arguments):
    input_images = []
    if arguments.get('image_ids'):
        input_images = list(map(int, arguments['image_ids'].split(',')))
    return get_classifier(arguments), input_images
    
    
def get_classifier(arguments):
    classifier_type = util.classifier_cli_to_constants(arguments['classifier'])
    if classifier_type == util.Constants.NearestNeighborClassifier:
        return nn.NN(arguments.get('m'))
    elif classifier_type == util.Constants.PersonalizedPageRankClassifier:
        return ppr.PPR(arguments.get('p'))
    return decision_trees.DecisionTree()

def execute_internal(classifier, query_image_ids, db, debug=False):
    start_time = time.time()

    fd = util.get_feature_model(TASK_ID)
    
    train_data = db.get_feature_descriptors(fd, train_data=True)
    train_labels_dict = db.get_id_label_dict(train_data=True)
    train_data_pairs = [(train_data[i], train_labels_dict[i]) for i in train_data]
    classifier.fit([i[0] for i in train_data_pairs], [i[1] for i in train_data_pairs])
    
    test_data = db.get_feature_descriptors(fd, train_data=False)
    test_labels = db.get_id_label_dict(train_data=False)
    input_data = get_data_to_be_predicted(test_data, query_image_ids)
    
    output_data = {"Image IDs":[], "Predicted Labels":[], "True Labels":[]}
    predictions = {}
    for image_id, image_vector in tqdm(input_data.items()):
        predicted_label = classifier.predict(image_vector)
        predictions[image_id] = predicted_label
        output_data["Image IDs"].append(image_id)
        output_data["Predicted Labels"].append(predicted_label)
        output_data['True Labels'].append(test_labels.get(image_id))

    util.AnalysePredictions(db, predictions)

    pathlib.Path(util.Constants.TASK_4_LOCATION).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_data).set_index("Image IDs").to_csv(os.path.join(util.Constants.TASK_4_LOCATION, 'output.csv'))

    end_time = time.time()
    if debug:
        print("Total time taken: ", end_time-start_time)


def get_data_to_be_predicted(data, input_ids):
    if not input_ids:
        return data
    return {i: data[i] for i in input_ids}
