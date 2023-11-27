from util import Constants
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torchvision

def visualizeKSimilarImages(query_image_id, k_similar_images, k_tags=[], output_filepath=""):
    imagenet_data = torchvision.datasets.Caltech101(root=Constants.CALTECH_DATASET_LOCATION, download=True)
    fig = plt.figure()
    plt.axis('off')
    c, r  = int(len(k_similar_images)//2) + (len(k_similar_images)%2), 3
    fig.add_subplot(r, c, 1)
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.imshow(imagenet_data[int(query_image_id)][0].resize((200,200)))
    plt.title("Query Image ID: "+str(query_image_id), fontsize = 10, pad = -2)
    for i in range(len(k_similar_images)):
        image_id = k_similar_images[i]
        cnt = c + i + 1
        fig.add_subplot(r, c, cnt)
        plt.axis('off')
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.imshow(imagenet_data[image_id][0].resize((200,200)))
        plt.title('id : ' + str(image_id), fontsize=10, pad=-2)
        if k_tags and k_tags[i]:
            plt.text(5, 220, k_tags[i])
    fig.suptitle("%d most similar images for image id: %d"%(len(k_similar_images), query_image_id), fontsize=16)
    if output_filepath:
        fig.savefig(output_filepath, bbox_inches="tight")
    plt.show()


def get_k_means_dr_mat(ls_data, feat_vector, k):
    mat_k = np.zeros(k)
    for i in range (k):
        mat_k[i] = np.linalg.norm(ls_data[i] - feat_vector)
    return mat_k

def get_lda_transform(lda_model, feat_vector):
    if len(np.array(feat_vector).shape) == 1:
        feat_vector = np.array(feat_vector).reshape(1, -1)
    if (np.array(feat_vector) < 0).any():
        # Normalize the input matrix using Min-Max scaling
        print("Using normalization as data contain negative value")
        min_max_scaler = MinMaxScaler()
        normalized_mat = min_max_scaler.fit_transform(np.array(feat_vector))
    else:
        # Use the original data if it doesn't contain negative values
        normalized_mat = np.array(feat_vector)
    return lda_model.transform(normalized_mat)[0]

def get_one_k_mat(ls_data, drt, k, feat_vector):

    if drt == Constants.SVD:
        # a k*4000 matrix
        V = ls_data[1][2]
        # feat_vec_ls = np.dot(feat_vector, V.reshape(V.shape[0], k))
        feat_vec_ls = np.dot(feat_vector, V.T)
    elif drt == Constants.LDA:
        # a model
        feat_vec_ls = get_lda_transform(ls_data[1][0], feat_vector)
    elif drt == Constants.NNMF:
        # a k*4000 matrix
        H = ls_data[1][1]
        feat_vec_ls = np.dot(feat_vector, H.T)
    elif drt == Constants.KMeans:
        # a k*4000 centroids matrix
        centroids = ls_data[1][0]
        feat_vec_ls = get_k_means_dr_mat(centroids, feat_vector, k)
    
    return feat_vec_ls

def get_imgs_k_mat(ls_data, drt):
    if drt == Constants.SVD :
        return ls_data[1][0]
    if drt == Constants.KMeans:
        return ls_data[1][1]
    elif drt == Constants.LDA:
        return ls_data[1][1]
    else:
        return ls_data[1][0]
    
def cosine_similarity(v1, v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))