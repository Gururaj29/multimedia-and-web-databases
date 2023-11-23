import os
import json
import pathlib
import glob
import numpy as np
from util import Constants
import pandas as pd

class Database:
    """Generic class to read and write from txt, np, json or csv files"""

    def __getpath(self):
        if self.path is not None:
            return self.path
        return os.getcwd()
    
    def __load__lbl_feat_vector_util(self,path) :
        feat_dict = {}
        for filename in os.listdir(path):
            if(filename[0] == '.') :
                continue
            with open(os.path.join(path,filename), 'r') as file:
                read_list = json.load(file)
                read_list = np.array(read_list).flatten()
                read_list = np.nan_to_num(read_list, nan=0.0)
                feat_dict[filename[:-4]] = read_list
                #label_dict[int(filename[:-4].split('_',1)[0])] = filename[:-4].split('_',1)[1] 
        return feat_dict   
    
    def __load_feat_vector_util(self,feat_path) :
        feat_dict = {}
        for filename in os.listdir(feat_path):
            if(filename[0] == '.') :
                continue
            with open(os.path.join(feat_path,filename), 'r') as file:
                read_list = json.load(file)
                read_list = np.array(read_list).flatten()
                read_list = np.nan_to_num(read_list, nan=0.0)
                feat_dict[int(filename[:-4].split('_',1)[0])] = read_list
        return feat_dict
    
    def __load_id_label_dict(self) :
        data = {}
        for filename in os.listdir(os.path.join(self.__getpath(), "Outputs","features", Constants.COLOR_MOMENTS)):
            if(filename[0] == '.') :
                continue
            data[int(filename[:-4].split('_',1)[0])] = filename[:-4].split('_',1)[1] 
        return data
    
    def __load_all_label_fds(self) :
        print('Loading Label Feature Vectors...')
        label_vector_location = os.path.join(self.__getpath(), "Outputs","features")
        return self.__load_all_fds(self.__load__lbl_feat_vector_util, label_vector_location)
    
    def __load_all_image_fds(self):
        print('Loading Image Feature Vectors...')
        feat_vector_location = os.path.join(self.__getpath(), "Outputs","features")
        return self.__load_all_fds(self.__load_feat_vector_util, feat_vector_location)
    
    def __load_all_fds(self, reader, path):
        data = {}
        data[Constants.COLOR_MOMENTS] = reader(os.path.join(path, Constants.COLOR_MOMENTS))
        data[Constants.HOG] = reader(os.path.join(path, Constants.HOG))
        data[Constants.ResNet_AvgPool_1024] = reader(os.path.join(path, Constants.ResNet_AvgPool_1024))
        data[Constants.ResNet_Layer3_1024] = reader(os.path.join(path, Constants.ResNet_Layer3_1024))
        data[Constants.ResNet_FC_1000] = reader(os.path.join(path, Constants.ResNet_FC_1000))
        data[Constants.ResNet_SoftMax_1000] = reader(os.path.join(path, Constants.ResNet_SoftMax_1000))
        return data

    def __load_similarity_matrices(self):
        data = {}
        path = os.path.join(self.__getpath(), 'Outputs', 'similarity_matrices')
        if os.path.exists(path):
            for file in glob.glob(path + "/*"):
                filename = os.path.basename(file)
                id, fd, _ = filename.split('-') # format is {image_image|label_label}-{fd}-similarity_matrix.csv
                data[(id, fd)] = pd.read_csv(file, index_col=0)
        return data
    
    def __init__(self, load_all_data = True, path = None):
        self.feature_descriptors = {}
        self.label_feature_descriptors = {}
        self.similarity_matrices = {}
        self.latent_semantics = {}
        self.id_label_dict = {} # format is {image_id: image_label}
        self.path = path
        if load_all_data:
            self.feature_descriptors = self.__load_all_image_fds()
            self.label_feature_descriptors = self.__load_all_label_fds()
            self.similarity_matrices = self.__load_similarity_matrices()
            self.id_label_dict = self.__load_id_label_dict()

    def get_feature_descriptors(self, fd):
        if fd not in self.feature_descriptors:
            return {}
        return self.feature_descriptors[fd]

    def get_label_feature_descriptors(self, fd):
        if fd not in self.label_feature_descriptors:
            return {}
        return self.label_feature_descriptors[fd]
    
    def get_id_label_dict(self):
        if not self.id_label_dict:
            return {}
        return self.id_label_dict
    
    def write_latent_semantics_into_file(self, ls, fd, drt, k, ids, latent_semantics_mat, data):
        # also update in internal dictionaries
        if ls not in self.latent_semantics:
            self.latent_semantics[ls] = {}
        self.latent_semantics[ls][(fd, drt, k)] = (ids, data)
        # create folder if doesn't exist already
        dir_path = os.path.join(self.__getpath(), 'Outputs', 'latent_semantics', ls)
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        filename = '%s_%s_%d.csv' % (fd, drt, k) # eg: hog_svd_5.csv
        file_path = os.path.join(dir_path, filename)
        # create dataframe and save to file
        # pd.DataFrame(latent_semantics_mat, index=ids).to_csv(file_path)
        pd.DataFrame(latent_semantics_mat).to_csv(file_path)
    
    def get_similarity_matrix(self, id, fd):
        # id is image_image or label_label
        return self.similarity_matrices[(id, fd)] if (id, fd) in self.similarity_matrices else None
    
    def write_similarity_matrix_to_file(self, id, fd, data, object_ids):
        # id is image_image or label_label
        df = pd.DataFrame(data, index=object_ids)
        self.similarity_matrices[(id, fd)] = df
        # create folder if doesn't exist already
        path = os.path.join(self.__getpath(), 'Outputs', 'similarity_matrices')
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # filename format is {image_image|label_label}-{fd}-similarity_matrix.csv
        df.to_csv(os.path.join(path, '%s-%s-similarity_matrix.csv'%(id, fd)))
    
    def get_latent_semantics_data(self, ls, fd, drt, k):
        if ls not in self.latent_semantics:
            return None
        key = (fd, drt, k)
        return self.latent_semantics[ls][key] if key in self.latent_semantics[ls] else None