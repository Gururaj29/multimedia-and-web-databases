import numpy as np
import pandas as pd
from Classifiers import multiclass_svm, svm

def Execute(arguments, db):
    img_id_tag, rfs = arguments.get('img_id_tag'), arguments.get('relevance_feedback_system')

    execute_internal(img_id_tag, rfs, db)

def execute_internal(img_id_tag, rfs, db, debug=False):

    '''
        Here, the image_id-tag pairs will be used to either 
        1. train an SVM, or
        2. rerank using the probabilistic model
        First, we need the image features, which we will pass to specific relevance feedback system (RFS)
    '''

    return