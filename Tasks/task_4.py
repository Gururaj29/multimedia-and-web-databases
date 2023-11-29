from Search import lsh
import util
from Tasks import task_util
from Classifiers import svm, multiclass_svm
import os
import pathlib
from RFS import ProbabilisticRelevanceFeedbackSystem
import numpy as np

TASK_ID = 4

RelevantTags = {util.Constants.Relevant, util.Constants.VeryRelevant}
IrrevantTags = {util.Constants.Irrelevant, util.Constants.VeryIrrelevant}

class QueryState:
    def __init__(self, query_image_id, k, feedback_system, query_image, query_k_best_results = [], query_results = set()):
        self.query_k_best_results = query_k_best_results
        self.query_results = query_results
        self.query_image_id = query_image_id
        self.image_tags = {}
        self.predicted_tags = {}
        self.irrelevant_images = set()
        self.predicted_irrelevant_images = set()
        self.k = k
        self.iter = 0 # search iterations
        self.tag_iter = 0 # tag iterations
        self.feedback_system = feedback_system # name of the feedback system being used
        self.feature_significance = list()
        self.query_image = query_image
    
    def setKBestResults(self, k_best_results):
        self.query_k_best_results = k_best_results
    
    def getImageTag(self, image_id):
        if image_id in self.image_tags:
            return self.image_tags[image_id]
        return self.predicted_tags.get(image_id)
    
    def setTags(self, image_ids, tags, predicted=False):
        if predicted:
            self.iter += 1
        else:
            self.tag_iter += 1
            self.iter = 1
        for i_id, tag in zip(image_ids, tags):
            if tag is not None:
                if predicted:
                    self.__setPredictedTag(i_id, tag)
                else:
                    self.__setUserProvidedTags(i_id, tag)
    
    def __setPredictedTag(self, image_id, tag):
        self.predicted_tags[image_id] = tag
        if tag in IrrevantTags:
            self.predicted_irrelevant_images.add(image_id)
    
    def __setUserProvidedTags(self, image_id, tag):
        self.image_tags[image_id] = tag
        if tag in IrrevantTags:
            self.irrelevant_images.add(image_id)

    def set_feature_significance(self, significance_list) :
        self.feature_significance = significance_list

    def get_feature_significance(self) :
        return self.feature_significance
    
    def getSearchExcludeSet(self):
        return self.irrelevant_images | self.predicted_irrelevant_images
    
    def getImagesAndTags(self):
        ''' returns two lists of image_ids and their tags'''
        images, tags = [], []
        for image_id, tag in self.image_tags.items():
            images.append(image_id)
            tags.append(tag)
        return images, tags
    
    def gotKRelevantImages(self):
        '''return a bool if there are at least k '''
        return len(self.query_results - self.getSearchExcludeSet()) >= self.k


def Execute(arguments, db):
    L, h, t, query_image_id, rfs_cli = arguments.get('L'), arguments.get('h'), arguments.get('t'), arguments.get('image_id'), arguments.get('rfs')
    rfs_type, rfs = get_relevence_feedback_system(rfs_cli)
    execute_internal(L, h, t, query_image_id, db, rfs_type, rfs)

def get_relevence_feedback_system(rfs_cli):
    rfs_type = util.rfs_cli_to_constants(rfs_cli)
    if rfs_type == util.Constants.SVMRelevanceFeedbackSystem:
        return rfs_type, multiclass_svm.MulticlassSVM(kernel="linear")
    elif rfs_type == util.Constants.ProbabilisticRelevanceFeedbackSystem:
        return rfs_type, ProbabilisticRelevanceFeedbackSystem.ProbabilisticRelevanceFeedbackSystem()
    return None, None

def execute_internal(L, h, t, query_image_id, db, rfs_type, rfs):
    fd = util.get_feature_model(TASK_ID)
    train_data = db.get_feature_descriptors(fd, train_data=True)
    query_image = db.get_feature_descriptors(fd, train_data=False).get(query_image_id)
    l = lsh.LSH(train_data, len(query_image), L, h)
    qs = QueryState(query_image_id, t, rfs_type, query_image)
    search = l.search(query_image, t)
    perform_lsh_search(search, qs)
    qs.setKBestResults(search.bestKResults(train_data))
    visualize_k_images(qs, save_file=True)
    if rfs_type:
        # run feedback loop if rfs_type is not None
        run_relevance_feedback_loop(rfs_type,rfs, qs, search, train_data)

def perform_lsh_search(lshSearch, qs):
    qs.query_results = lshSearch.search(qs.getSearchExcludeSet())
    if qs.iter != 0:
        print("-"*50)
        print("Running search iteration number: %d"%qs.iter)
    lshSearch.printSearchAnalytics()

def run_relevance_feedback_loop(rfs_type,rfs, qs, search, data):
    tagging_prompt(qs, qs.query_k_best_results)
    images, tags = qs.getImagesAndTags() # for RFS models
    if rfs_type == util.Constants.SVMRelevanceFeedbackSystem :
        rfs.fit([data[image] for image in images], tags)
    else :
        rfs.fit(qs.query_image, np.array([data[image] for image in images]), tags)
        qs.set_feature_significance(rfs.get_significance())
    while not qs.gotKRelevantImages() and qs.iter <= 100: # kept limit on the number of iterations for now
        perform_lsh_search(search, qs)
        predict_tags(qs, rfs, data)
    if rfs_type == util.Constants.SVMRelevanceFeedbackSystem :
        qs.setKBestResults(search.bestKResults(data, qs.getSearchExcludeSet()))
    else :
        qs.setKBestResults(search.bestKResults_Probabilistic(data, qs.get_feature_significance(),qs.getSearchExcludeSet()))
    visualize_k_images(qs, save_file=True)
    if boolean_prompt("Want to retag images"):
        run_relevance_feedback_loop(rfs_type, rfs, qs, search, data)

def predict_tags(qs, rfs, data):
    image_ids_with_updated_tags = []
    tags = []
    for image_id in qs.query_results:
        if not qs.getImageTag(image_id):
            if(qs.feedback_system == util.Constants.SVMRelevanceFeedbackSystem) :
                tag = rfs.predict(data[image_id])
            # print("Tag predicted for image-id " + str(image_id) + ": " + tag)
                image_ids_with_updated_tags.append(image_id)
                tags.append(tag)
    qs.setTags(image_ids_with_updated_tags, tags, predicted = True)

def visualize_k_images(qs, save_file=False, tags=[]):
    output_filepath = ""
    if save_file:
        filename = "search_output_%s_%d_%d_%d.png"%(qs.feedback_system, qs.query_image_id, qs.k, qs.tag_iter) if qs.feedback_system else "search_output_%d_%d.png"%(qs.query_image_id, qs.k)
        output_filepath = os.path.join(util.Constants.TASK_4_LOCATION, filename)
        pathlib.Path(util.Constants.TASK_4_LOCATION).mkdir(parents=True, exist_ok=True)
    task_util.visualizeKSimilarImages(qs.query_image_id, qs.query_k_best_results, tags, output_filepath)

# method for the input prompts
def format_print_message_for_task(message):
    return "\x1b[32mtask_5| %s : \x1b[0m"%message

def boolean_prompt(message):
    inp = input(format_print_message_for_task("%s (Y/n)?"%message))
    return inp == "Y" or inp == "y" or inp == "yes" or inp == "Yes"

def tagging_prompt(qs, image_ids):
    input_to_constants_map = {
        "R": util.Constants.Relevant,
        "VR": util.Constants.VeryRelevant,
        "IR": util.Constants.Irrelevant,
        "VIR": util.Constants.VeryIrrelevant,
        "": None
    }
    print(format_print_message_for_task("Tag output images based on their relevance {R=Relevant, VR=Very Relevant, IR=Irrevelant, VIR = Very Irrelevant or None}"))
    tags = []
    for image_id in image_ids:
        tag_input = input(format_print_message_for_task("Enter a tag for image ID - %d"%image_id))
        while tag_input not in input_to_constants_map:
            tag_input = input(format_print_message_for_task("Please enter a valid tag for image ID - %d"%image_id))
        tags.append(input_to_constants_map[tag_input])
    qs.setTags(image_ids, tags)

    showTags = boolean_prompt("Show confirmation")
    if showTags:
        visualize_k_images(qs, tags=tags)
