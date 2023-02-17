import numpy as np
import cv2
import os
from scipy.spatial.distance import euclidean
import skimage.filters as filters
import pickle
import mtcnn

def normalize(v):
    len = np.linalg.norm(v, axis=-1).reshape(-1, 1)
    return v / len

def save_data(data, path):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def load_data(path):
    pickle_in = open(path,"rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data

def align_face(img, face_detector, resolution):
    eye_pos_w, eye_pos_h = 0.28, 0.28
    width, height = resolution, resolution
    img_rgb = img[..., ::-1]

    results = face_detector.detect_faces(img_rgb)
    result = results[0]

    l_e = result['keypoints']['left_eye']
    r_e = result['keypoints']['right_eye']
    center = (((r_e[0] + l_e[0]) // 2), ((r_e[1] + l_e[1]) // 2))

    dx = (r_e[0] - l_e[0])
    dy = (r_e[1] - l_e[1])
    dist = euclidean(l_e, r_e)

    angle = np.degrees(np.arctan2(dy, dx)) + 360
    scale = width * (1 - (2 * eye_pos_w)) / dist

    tx = width * 0.5
    ty = height * eye_pos_h

    m = cv2.getRotationMatrix2D(center, angle, scale)

    m[0, 2] += (tx - center[0])
    m[1, 2] += (ty - center[1])

    face_align = cv2.warpAffine(img, m, (width, height))

    return face_align

def write_faces(source, dest, resolution=100, is_correct_lighting=False, is_blur=False):
    face_detector = mtcnn.MTCNN()
    labels = os.listdir(source)
    for label in labels:
        img = cv2.imread(os.path.join(source, label))
        img = align_face(img, face_detector, resolution=resolution)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (resolution, resolution))
        if is_correct_lighting:
            img = correct_lighting(img)
        if is_blur:
            img = blur(img)
        cv2.imwrite(os.path.join(dest, label), img)

def blur(img):
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    return blurred

def sharpen(img):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    sharpened = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return sharpened

def contrast(img):
    contrasted = cv2.convertScaleAbs(img, alpha=1.5, beta=-127.5)
    return contrasted

def correct_lighting(img):
    smooth = cv2.GaussianBlur(img, (33,33), 0)
    division = cv2.divide(img, smooth, scale=255)
    sharp = filters.unsharp_mask(division, radius=1, amount=0.5, channel_axis=True, preserve_range=False)
    sharp = (255*sharp).clip(0,255).astype(np.uint8)
    return sharp

def get_data(path, resolution=100, crop_align=False):
    labels = os.listdir(path)
    vectors = []
    images = []
    for label in labels:
        if crop_align:
            img = cv2.imread(os.path.join(path, label))
            img = align_face(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(os.path.join(path, label), cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (resolution, resolution))
        images.append(img)
        vectors.append(img.flatten())
    return np.array(images), np.array(vectors), np.array(labels)

if __name__ == "__main__":
    is_correct_lighting = True
    is_blur = False
    resolution = 100
    write_faces("raw_train", "train", is_correct_lighting=is_correct_lighting, is_blur=blur, resolution=resolution)
    write_faces("raw_test", "test", is_correct_lighting=is_correct_lighting, is_blur=blur, resolution=resolution)
    write_faces("raw_dif", "dif", is_correct_lighting=is_correct_lighting, is_blur=blur, resolution=resolution)
    print("Done.")