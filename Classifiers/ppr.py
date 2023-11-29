import os
import ast
import heapq
import random
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

import util

class PPR:
    '''
        class for personalized pagerank classifier
    '''
    def __init__(self, random_jump_probability = 0.15) -> None:
        self.random_jump_probability = random_jump_probability
        self.num_walks = 1000000
        
        self.label_features = self.get_label_dict("./Outputs/train/labels/resnet_fc_1000")
        self.training_images_features = self.get_features_dict("./Outputs/train/features/resnet_fc_1000")
        self.testing_images_features = self.get_features_dict("./Outputs/test/features/resnet_fc_1000")

        self.graph = self.create_graph(self.training_images_features, self.label_features)
        self.predictions = {}
        for image_id, features in self.testing_images_features.items():
            image_node = "image#" + str(image_id)
            image_features = np.array(features)
            image_features_2d = image_features.reshape(1, -1)

            self.graph.add_node(image_node)
            
            self.seeds = self.get_seeds(self.graph, image_node, image_features_2d, self.label_features)[:1]
            # if self.random_jump_probability < 0.5:
            #     self.seeds = self.seeds[:1]
            # else:
            #     self.seeds = self.seeds[:1]

            pagerank = self.personalized_page_rank(graph=self.graph, seeds=self.seeds, source=image_node, num_walks=self.num_walks, beta=self.random_jump_probability)
            
            labels_probabilities = {node.split('#')[1]: score for node, score in pagerank.items() if node.startswith('label#')}
            predicted_labels = max(labels_probabilities, key=labels_probabilities.get)
            self.graph.remove_node(image_node)
            
            self.predictions[image_id] = predicted_labels
    
    def get_seeds(self, graph, image_node, image_features, label_features):
        seeds = {}
        for label_name, label_features in label_features.items():
            label_node = "label#" + label_name
            label_features = np.array(label_features) 
            label_features_2d = label_features.reshape(1, -1) 

            similarity_score = cosine_similarity(image_features, label_features_2d)[0][0]

            graph.add_edge(image_node, label_node, weight=similarity_score)
            seeds[label_node] = similarity_score
        return sorted(seeds, key=lambda x:seeds[x], reverse=True)
        

    def get_features_dict(self, folder_path):
        file_extension = ".txt"

        files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]
        data = {}
        for file in files:
            image_id = file.split("_")[0]
            
            with open(os.path.join(folder_path, file), 'r') as f:
                file_content = f.readline().strip()  
                file_data = ast.literal_eval(file_content)
                data[int(image_id)] = file_data
        
        return data
    
    def get_label_dict(self, folder_path):
        file_extension = ".txt"

        files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]
        data = {}

        for file in files:
            label = file.split(".")[0]
            
            with open(os.path.join(folder_path, file), 'r') as f:
                file_content = f.readline().strip()  
                file_data = ast.literal_eval(file_content)
                data[label] = file_data
        
        return data

    def create_graph(self, training_images_features, label_features):

        graph = nx.Graph()

        for image_id, image_features in training_images_features.items():
            # Add nodes for even images
            image_node = "image#" + str(image_id)
            graph.add_node(image_node)
            
            # Add edges between even images and their corresponding labels
            image_features = np.array(image_features)
            image_features_2d = image_features.reshape(1, -1)
            
            for label_name, label_feature in label_features.items():
                label_node = "label#" + label_name
                label_feature = np.array(label_feature) 
                label_features_2d = label_feature.reshape(1, -1)
                
                similarity_score = cosine_similarity(image_features_2d, label_features_2d)[0][0]
                
                graph.add_node(label_node)
                graph.add_edge(image_node, label_node, weight=similarity_score)
        
        return graph
        

    def personalized_page_rank(self, graph, seeds, source, num_walks, beta):
        # 1. initialize pi to zeros
        counts = {node: 0 for node in graph.nodes()}

        # 4. repeat for a number of walks
        for i in range(num_walks):
            # 3. store visit count
            counts[source] += 1

            # 2. get a random float between 0 and 1
            prob = random.random()
            # 2.1 go to any connected node with probability (1-beta)
            if prob < (1 - beta):
                source = random.choice(list(graph[source]))
            # 2.2 go to a seed node with probability (beta)
            else:
                source = random.choice(seeds)

        # normalizing
        for node, visits in counts.items():
            counts[node] = visits / num_walks

        # return sorted (descending)
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    
    def predict(self, image_id):
        return self.predictions[image_id]