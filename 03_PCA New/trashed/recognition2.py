import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

resolution = 200

def save_data(data, path):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def load_data(path):
    pickle_in = open(path,"rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data

def train():
    path = "train_images2"

    labels = os.listdir(path)
    full_labels = [os.path.join(path, label) for label in labels]

    faces = []

    for image in full_labels:
        face_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        face_image = cv2.resize(face_image, (resolution, resolution))
        faces.append(face_image)
    
    # fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
    # faceimages = faces[-16:]
    # for i in range(16):
    #     ax = axes[i % 4][i // 4]
    #     ax.axis("off")
    #     ax.imshow(faceimages[i], cmap="gray")
    # plt.show()
    
    facematrix = []
    for face in faces:
        facematrix.append(face.flatten())
    facematrix = np.array(facematrix)

    
    pca = PCA().fit(facematrix)
    n_components = 30
    eigenfaces = pca.components_[:n_components]
    
    # fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
    # for i in range(16):
    #     ax = axes[i % 4][i // 4]
    #     ax.axis("off")
    #     ax.imshow(eigenfaces[i].reshape(resolution, resolution), cmap="gray")
    # plt.show()
    
    weights = eigenfaces @ (facematrix - pca.mean_).T
    save_data(labels, os.path.join("model2", "labels.pickle"))
    save_data(pca, os.path.join("model2", "pca.pickle"))
    save_data(eigenfaces, os.path.join("model2", "eigenfaces.pickle"))
    save_data(weights, os.path.join("model2", "weights.pickle"))

    print("Model successfully saved.")

def predict(query):

    labels = load_data(os.path.join("model2", "labels.pickle"))
    pca = load_data(os.path.join("model2", "pca.pickle"))
    eigenfaces = load_data(os.path.join("model2", "eigenfaces.pickle"))
    weights = load_data(os.path.join("model2", "weights.pickle"))

    query = cv2.resize(query, (resolution, resolution))
    query = query.reshape(1, -1)
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)

    # sorted_indices = np.argsort(euclidean_distance)
    # for i in range(5):
    #     rank = sorted_indices[i]
    #     print(labels[rank].split(".")[0], euclidean_distance[rank], end="\n")

    # fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    # axes[0].imshow(query.reshape((resolution, resolution)), cmap="gray")
    # axes[0].set_title("Query")
    # axes[1].imshow(facematrix[best_match].reshape(resolution, resolution), cmap="gray")
    # axes[1].set_title("Best match")
    # plt.show()

    return labels[best_match].split(".")[0], euclidean_distance[best_match]

if __name__ == "__main__":
    train()

# query = cv2.imread(os.path.join("test_images", "jerome.jpg"), cv2.IMREAD_GRAYSCALE)
# predict(query)
