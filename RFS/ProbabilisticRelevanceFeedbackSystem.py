import numpy as np
import math
from util import Constants

class ProbabilisticRelevanceFeedbackSystem :
    def __init__(self ) :
        self.significance = list()
    
    def fit(self,query, fs_, tags) :
        # query = query vector 
        # fs_ = feature vectors of images
        # tags = tags of images 
        
        feats = fs_.copy()
        q = query.copy()
        threshold_b = self.make_binary(feats)
        for i in range(len(q)) :
            if(q[i] >= threshold_b[i]) :
                q[i] = 1
            else :
                q[i] = 0
        
        T = self.find_threshold(feats,q)
        tagset = [Constants.Relevant,Constants.VeryRelevant,Constants.Irrelevant,Constants.VeryIrrelevant] #make it constants
        im_tag_dict = {}
        for tag in tagset :
            im_t = []
            # get images in that tagset
            for j in range(len(feats)) :
                if(tags[j] == tag) :
                    im_t.append(feats[j])
            im_tag_dict[tag] = im_t
        
        significance = list()
        
#         print(feats)
        
        for i in range(len(feats[0])) :
            # calculate significance of ith feature
            
            # now for each tag find set of images that follows these conditions 
            # b = similarity(f,q) > T without j where fj = 1
            # c = similarity(f,q) > T without j where fj = 0
            # remove ith column from feature 
            tag_importance = {}
            
            for tag in tagset :
                    new_query = np.delete(np.array(q),i)
                    b = 0.01 # to remove devide 0 issues
                    c = 0.01
                    for j in range(len(im_tag_dict[tag])) :
                        feature_vector = np.array(im_tag_dict[tag][j])
                        f_j = feature_vector[i]
                        feature_vector = np.delete(feature_vector,i)
                        if(f_j == 1 and (np.dot(feature_vector,new_query) / ((np.linalg.norm(feature_vector)*np.linalg.norm(new_query)+0.01))) > T) :
                            b+=1
                        if(f_j == 0 and (np.dot(feature_vector,new_query) / ((np.linalg.norm(feature_vector)*np.linalg.norm(new_query)+0.01))) > T) :
                            c+=1
                    tag_importance[tag] = b/c
            significance.append(math.log10((tag_importance[Constants.VeryRelevant]**2 * tag_importance[Constants.Relevant]) / (tag_importance[Constants.VeryIrrelevant]**2 * tag_importance[Constants.Irrelevant])))
        # s = np.array(significance)
#         print(list((s - np.min(s))/(np.max(s) - np.min(s))))
        self.significance = significance
        
            
    def make_binary(self,fs_) :
        fs_ = fs_.T
        thr_ = list()
        for l in fs_ :
            T = np.percentile(l, 50)
#             print(T)
            thr_.append(T)
            for i in range(len(l)) :
                if(l[i] >= T) : 
                    l[i] = 1
                else : 
                    l[i] = 0
        return thr_
    
    def get_significance(self) :
        return self.significance
    
    def find_threshold(self,feats,q) :
        return min( [np.dot(f,q)/((np.linalg.norm(f)*np.linalg.norm(q))+0.01) for f in feats] )
    
        
        