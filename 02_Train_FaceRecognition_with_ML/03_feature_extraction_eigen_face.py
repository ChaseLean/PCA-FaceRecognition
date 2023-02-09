import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

# Eigen Face
from sklearn.decomposition import PCA

import pickle
data = pickle.load(open('./data/data_images_100_100.pickle',
                   mode='rb'))  # load the data
data.head()

# Mean Face
X = data.drop('gender', axis=1).values  # all images
X
mean_face = X.mean(axis=0)  # flatten mean face
mean_face.shape

# visualize mean face
plt.imshow(mean_face.reshape((100, 100)), cmap='gray')
plt.axis('off')
plt.show()

X_t = X - mean_face  # transformed data

# Apply X_t data to PCA
pca = PCA(n_components=None, whiten=True, svd_solver='auto')
pca.fit(X_t)

exp_var_df = pd.DataFrame()
exp_var_df['explained_var'] = pca.explained_variance_ratio_
exp_var_df['cum_explained_var'] = exp_var_df['explained_var'].cumsum()
exp_var_df['principal_components'] = np.arange(1, len(exp_var_df)+1)

exp_var_df.head()

exp_var_df.set_index('principal_components', inplace=True)

# visualize explained variance
fig, ax = plt.subplots(nrows=2, figsize=(15, 12))

exp_var_df['explained_var'].head(100).plot(kind='line', marker='o', ax=ax[0])
exp_var_df['cum_explained_var'].head(100).plot(
    kind='line', marker='o', ax=ax[1])

pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto')
pca_data = pca_50.fit_transform(X_t)

pca_data.shape

# saving data and models
y = data['gender'].values  # independent variables
np.savez('./data/data_pca_50_target', pca_data, y)

# saving the model
pca_dict = {'pca': pca_50, 'mean_face': mean_face}

pickle.dump(pca_dict, open('model/pca_dict.pickle', 'wb'))

# Visualize Eigen Image
pca_data_inv = pca_50.inverse_transform(pca_data)

pca_data_inv.shape

eig_img = pca_data_inv[0, :].reshape((100, 100))
eig_img.shape

plt.imshow(eig_img, cmap='gray')
plt.axis('off')

np.random.seed(1001)
pics = np.random.randint(0, 4319, 40)
plt.figure(figsize=(15, 8))
for i, pic in enumerate(pics):
    plt.subplot(4, 10, i+1)
    img = X[pic:pic+1].reshape(100, 100)
    plt.imshow(img, cmap='gray')
    plt.title('{}'.format(y[pic]))
    plt.xticks([])
    plt.yticks([])
plt.show()

print("="*20+'Eigen Images'+"="*20)
plt.figure(figsize=(15, 8))
for i, pic in enumerate(pics):
    plt.subplot(4, 10, i+1)
    img = pca_data_inv[pic:pic+1].reshape(100, 100)
    plt.imshow(img, cmap='gray')
    plt.title('{}'.format(y[pic]))
    plt.xticks([])
    plt.yticks([])

plt.show()
