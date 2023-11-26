from util import Constants
import matplotlib.pyplot as plt
import torchvision

def visualizeKSimilarImages(query_image_id, k_similar_images):
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
    fig.suptitle("%d most similar images for image id: %d"%(len(k_similar_images), query_image_id), fontsize=16)
    plt.show()