import math
import numpy as np
import cv2
import os
import pickle
from matplotlib import pyplot as plt

def train():
    resolution = 200
    # Sets the resolution of the image
    total_pixels = resolution ** 2

    # The folder name for the training data
    path = "train_images"

    # Gets all the images by their name in the folder
    labels = os.listdir(path)
    save_data(labels, os.path.join("model", "labels.pickle"))
    full_labels = [os.path.join(path, label) for label in labels]

    # A list containing all the faces
    face_vector = []

    # For each face, reshape it into a vector and append it to the list
    for image in full_labels:
        face_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        face_image = cv2.resize(face_image, (resolution, resolution))
        face_image = face_image.reshape(total_pixels,)
        face_vector.append(face_image)

    face_vector = np.asarray(face_vector)
    face_vector = face_vector.transpose()

    # Calculate the average face
    avg_face_vector = face_vector.mean(axis=1)

    # Show the picture of the average face
    plt.title("Average face")
    plt.imshow(avg_face_vector.reshape(resolution, resolution), cmap="gray")
    plt.show()

    # Reshape the average face into a vector
    avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)

    # Normalize each face in the face array by subtracting it from the average face
    normalized_face_vector = face_vector - avg_face_vector

    # Calculate the covariance matrix and the eigenvalues
    covariance_matrix = np.cov(np.transpose(normalized_face_vector))
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Keep the k largest eigenvectors, and generate k eigenfaces
    k = 15
    k_eigen_vectors = eigen_vectors[0:k, :]
    eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
    weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))
    save_data(avg_face_vector, os.path.join("model", "average_face_vector.pickle"))
    save_data(eigen_faces, os.path.join("model", "eigen_faces.pickle"))
    save_data(weights, os.path.join("model", "weights.pickle"))

    print("Finished training. Model information has been saved.")

def save_data(data, path):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def load_data(path):
    pickle_in = open(path,"rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data

def predict(img):
    resolution = 200
    total_pixels = resolution ** 2

    labels = load_data(os.path.join("model", "labels.pickle"))
    avg_face_vector = load_data(os.path.join("model", "average_face_vector.pickle"))
    eigen_faces = load_data(os.path.join("model", "eigen_faces.pickle"))
    weights = load_data(os.path.join("model", "weights.pickle"))

    img = cv2.resize(img, (resolution, resolution))

    # Reshape, rescale and normalize the test image
    img = img.reshape(total_pixels, 1)
    test_normalized_face_vector = img - avg_face_vector
    test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces)).reshape(-1)

    # Find the index inside the train data that best matches the test data, and print the index
    index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))

    distance = np.linalg.norm(test_weight - weights, axis=1)
    min_distance = distance[index]
    confidence = 100
    if min_distance != 0:
        confidence = math.exp(- min_distance * 1e-8) * 100
    return labels[index].split(".")[0], confidence

if __name__ == "__main__":
    train()

# face = cv2.imread(os.path.join("test_images", "hengtest.jpg"), cv2.IMREAD_GRAYSCALE)
# name, confidence = predict(face)
# print(name, confidence)