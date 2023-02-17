# https://melaniesoek0120.medium.com/principal-component-analysis-pca-facial-recognition-5e1021f55151

# Import necessary packages
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from processing import *

def plot_explained_variance(pca, n):
    plt.plot(range(1, n + 1), pca.explained_variance_ratio_.cumsum())
    plt.title('Explained Variance')
    plt.xlabel('Number of Principle Components')
    plt.show()

def show_avg_face(pca):
    plt.imshow(pca.mean_.reshape(*resolution), cmap="gray")
    plt.show()

def show_eigenfaces(pca, n, resolution):
    plot_dim = math.floor(math.sqrt(n))
    fig, axes = plt.subplots(plot_dim,plot_dim,sharex=True,sharey=True,figsize=(8,10))
    for i in range(plot_dim ** 2):
        ax = axes[i % plot_dim][i // plot_dim]
        ax.axis("off")
        ax.imshow(pca.components_[i].reshape(*resolution), cmap="gray")
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=model.classes_, normalize='true')
    plt.show()

def print_errors(model, X_test, y_test):
    for a, b in zip(model.predict(X_test), y_test):
        if a != b:
            print( 'Predicted:{:<10s} Actual:{:<10s}'.format(a, b))

images_train, x_train, y_train = get_data("train")
images_test, x_test, y_test = get_data("dif")
y_train = [name[name.find("[")+1:name.find("]")] for name in y_train]
y_test = [name[name.find("[")+1:name.find("]")] for name in y_test]

# images, x, y = get_data("dif")
# y = [name[name.find("[")+1:name.find("]")] for name in y]
# x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.4)

unique_names = np.unique(np.array(y_train))
resolution = (100, 100)

n = 30
pca = PCA(n_components=n, whiten=True)
X_train = pca.fit_transform(x_train)
X_test = pca.transform(x_test)

plot_explained_variance(pca, n)
show_avg_face(pca)
show_eigenfaces(pca, n, resolution)

model = svm.LinearSVC()
model.fit(X_train, y_train)

accuracy_train= model.score(X_train, y_train)
accuracy_test= model.score(X_test, y_test)
print("Classifier: Linear SVM")
print('Accuracy - train data: {}'.format(accuracy_train))
print('Accuracy - test data : {}'.format(accuracy_test))
plot_confusion_matrix(model, X_test, y_test)

save_data(pca, os.path.join("models", "pca.pickle"))
save_data(model, os.path.join("models", "svm.pickle"))

model = neighbors.KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
accuracy_train= model.score(X_train, y_train)
accuracy_test= model.score(X_test, y_test)
print("Classifier: KNN")
print('Accuracy - train data: {}'.format(accuracy_train))
print('Accuracy - test data : {}'.format(accuracy_test))
plot_confusion_matrix(model, X_test, y_test)

