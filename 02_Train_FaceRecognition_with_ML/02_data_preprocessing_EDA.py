import pickle
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# computer vision library
import cv2

# glob
from glob import glob

import warnings
warnings.filterwarnings('ignore')

# extract path of male and female in crop_data folder and put them in a list
fpath = glob('./crop_data/female/*.jpg')
mpath = glob('./crop_data/male/*.jpg')

df_female = pd.DataFrame(fpath, columns=['filepath'])
df_female['gender'] = 'female'

df_male = pd.DataFrame(mpath, columns=['filepath'])
df_male['gender'] = 'male'

df = pd.concat((df_female, df_male), axis=0)

df.head()

df.tail()

df.shape

# it will take each image path
# then return width of the image


def get_size(path):
    img = cv2.imread(path)
    return img.shape[0]


# store dimension of image in this columns
df['dimension'] = df['filepath'].apply(get_size)

df.head()

dist_gender = df['gender'].value_counts()
dist_gender

fig, ax = plt.subplots(nrows=1, ncols=2)
dist_gender.plot(kind='bar', ax=ax[0])
dist_gender.plot(kind='pie', ax=ax[1], autopct='%0.0f%%')
plt.show()

# What Distribution of size of all Images
# Histogram
# Box Plot
# Split by “Gender”
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
sns.histplot(df['dimension'])
plt.subplot(2, 1, 2)
sns.boxplot(df['dimension'])
plt.show()

sns.catplot(data=df, x='gender', y='dimension', kind='box')

df_filter = df.query('dimension > 60')
df_filter.shape

df_filter['gender'].value_counts(normalize=True)


# Structure the image¶
# 100 x 100
def structuring(path):
    try:

        # step - 1: read image
        img = cv2.imread(path)  # BGR
        # step - 2: convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # step -3: resize into 100 x 100 array

        size = gray.shape[0]
        if size >= 100:
            # cv2.INTER_AREA (SHINK)
            gray_resize = cv2.resize(gray, (100, 100), cv2.INTER_AREA)
        else:
            # cv2.INTER_CUBIC (ENLARGE)
            gray_resize = cv2.resize(gray, (100, 100), cv2.INTER_CUBIC)

        # step -4: Flatten Image (1x10,000)
        flatten_image = gray_resize.flatten()
        return flatten_image

    except:
        return None


df_filter['data'] = df_filter['filepath'].apply(
    structuring)  # convert all images into 100 x 100
df_filter.head()

data = df_filter['data'].apply(pd.Series)
data.columns = [f"pixel_{i}" for i in data.columns]
data.head()

# since for 8 bit image max value is 255
# therefore we are dividing each and every pixel with 255
data = data/255.0
data['gender'] = df_filter['gender']
data.head()

###
data.isnull().sum().sum()

# remove the missing values
data.dropna(inplace=True)

data.shape

pickle.dump(data, open('./data/data_images_100_100.pickle', mode='wb'))
