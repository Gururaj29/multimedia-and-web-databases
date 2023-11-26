from Search import lsh
from util import Constants
from Tasks import task_util

def Execute(arguments, db):
    L, h, t, query_image_id = arguments.get('L'), arguments.get('h'), arguments.get('t'), arguments.get('image_id')
    execute_internal(L, h, t, query_image_id, db)

def execute_internal(L, h, t, query_image_id, db):
    fd = Constants.ResNet_FC_1000
    train_data = db.get_feature_descriptors(fd, train_data=True)
    query_image = db.get_feature_descriptors(fd, train_data=False).get(query_image_id)
    l = lsh.LSH(train_data, len(query_image), L, h)
    perform_lsh_search(train_data, l.search(query_image, t), query_image_id)

def perform_lsh_search(data, lshSearch, query_image_id):
    task_util.visualizeKSimilarImages(query_image_id, lshSearch.bestKResults(data))
    lshSearch.printSearchAnalytics()