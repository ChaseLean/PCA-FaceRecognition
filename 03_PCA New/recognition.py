import math
import numpy as np
import cv2
import os
import pickle
from matplotlib import pyplot as plt
import align

def normalize(v):
    len = np.linalg.norm(v, axis=-1).reshape(-1, 1)
    return v / len

def train(resolution=200):
    # Sets the resolution of the image
    total_pixels = resolution ** 2

    # The folder name for the training data
    path = "train_images_aligned"

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
        face_image = (face_image - np.mean(face_image)) / np.linalg.norm(face_image)
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

    # Sorts the eigenvalues in descending order
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = normalize(eigen_vectors[sorted_indices])

    # Calculate the explained variance by determining the ratio of each eigenvalue to the sum of all eigenvalues
    sum_eig = sum(eigen_values)
    explained_variance = [i/sum_eig for i in eigen_values]
    plt.plot(range(1, len(eigen_values) + 1), explained_variance)
    plt.show()

    # Keep the k largest eigenvectors, and generate k eigenfaces
    k = 20
    k_eigen_vectors = eigen_vectors[0:k, :]
    eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
    weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))

    # Plots the first few eigenfaces
    plot_dim = math.floor(math.sqrt(k))
    fig, axes = plt.subplots(plot_dim,plot_dim,sharex=True,sharey=True,figsize=(8,10))
    for i in range(plot_dim ** 2):
        ax = axes[i % plot_dim][i // plot_dim]
        ax.axis("off")
        ax.imshow(eigen_faces[i].reshape(resolution, resolution), cmap="gray")
    plt.show()

    # Saves data into a file, so that the training only needs to be carried out once
    save_data(avg_face_vector, os.path.join("model", "average_face_vector.pickle"))
    save_data(eigen_faces, os.path.join("model", "eigen_faces.pickle"))
    save_data(weights, os.path.join("model", "weights.pickle"))

    print("Finished training. Model information has been saved.")

# Saves data using pickle
def save_data(data, path):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

# Loads data using pickle
def load_data(path):
    pickle_in = open(path,"rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data

# Finds the best match of a new face based on the training data
def predict(img, resolution=200):
    total_pixels = resolution ** 2

    # Loads the information that was previously trained
    labels = load_data(os.path.join("model", "labels.pickle"))
    avg_face_vector = load_data(os.path.join("model", "average_face_vector.pickle"))
    eigen_faces = load_data(os.path.join("model", "eigen_faces.pickle"))
    weights = load_data(os.path.join("model", "weights.pickle"))

    img = cv2.resize(img, (resolution, resolution))

    # Reshape, rescale and normalize the test image
    img = img.reshape(total_pixels, 1)
    img = (img - np.mean(img)) / np.linalg.norm(img)
    test_normalized_face_vector = img - avg_face_vector
    test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces)).reshape(-1)

    # Find the index inside the train data that best matches the test data, and print the index
    distance = np.linalg.norm(test_weight - weights, axis=1)
    index =  np.argmin(distance)
    sorted_indices = np.argsort(distance)

    # Prints the top 5 matches and their respective distances
    print("-" * 20)
    for i in range(5):
        rank = sorted_indices[i]
        print(labels[rank], distance[rank], end="\n")
    print("-" * 20)

    # Returns the name of the best match, and the respective distance
    min_distance = distance[index]
    name = labels[index].strip(".jpg")
    return name, min_distance

if __name__ == "__main__":
    # First train the model (you only need to train it once)
    train()

    # Read the test face, and then find the closest match
    face = cv2.imread(os.path.join("test_images", "room.jpg"))
    face = align.align(face)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    name, distance = predict(gray)
    print(name, distance)