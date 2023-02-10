# Source: https://github.com/serengil/tensorflow-101/blob/master/python/face-alignment.py

import os
import cv2
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from PIL import Image

#------------------------

def euclidean_distance(a, b):
	x1 = a[0]; y1 = a[1]
	x2 = b[0]; y2 = b[1]
	
	return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectFace(img):
	face_detector, eye_detector = get_detector()
	faces = face_detector.detectMultiScale(img, 1.1, 5)

	print(len(faces), "test")

	if len(faces) > 0:
		face = faces[0]
		face_x, face_y, face_w, face_h = face
		img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img_gray
	else:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img_gray

def alignFace(img_path):
	face_detector, eye_detector = get_detector()
	img = cv2.imread(img_path)

	print(img_path)
	img_raw = img.copy()
	img, gray_img = detectFace(img)
	eyes = eye_detector.detectMultiScale(gray_img, 1.1, 5)
	if len(eyes) >= 2:
		base_eyes = eyes[:, 2]
	else:
		base_eyes = eyes	
	items = []
	for i in range(0, len(base_eyes)):
		item = (base_eyes[i], i)
		items.append(item)
	df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)

	eyes = eyes[df.idx.values[0:2]]
	
	eye_1 = eyes[0]; eye_2 = eyes[1]
	
	if eye_1[0] < eye_2[0]:
		left_eye = eye_1
		right_eye = eye_2
	else:
		left_eye = eye_2
		right_eye = eye_1
	
	left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
	left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
	
	right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
	right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
	
	cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
	cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
	
	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1
	
	cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
	
	cv2.line(img,right_eye_center, left_eye_center,(67,67,67),1)
	cv2.line(img,left_eye_center, point_3rd,(67,67,67),1)
	cv2.line(img,right_eye_center, point_3rd,(67,67,67),1)
	
	a = euclidean_distance(left_eye_center, point_3rd)
	b = euclidean_distance(right_eye_center, point_3rd)
	c = euclidean_distance(right_eye_center, left_eye_center)
	
	cos_a = (b*b + c*c - a*a)/(2*b*c)
	angle = np.arccos(cos_a)
	
	angle = (angle * 180) / math.pi
	if direction == -1:
		angle = 90 - angle
	
	new_img = Image.fromarray(img_raw)
	new_img = np.array(new_img.rotate(direction * angle))
	
	return new_img

def get_detector():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]

	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
	eye_detector_path = path+"/data/haarcascade_eye.xml"

	face_detector = cv2.CascadeClassifier(face_detector_path)
	eye_detector = cv2.CascadeClassifier(eye_detector_path)
	return face_detector, eye_detector

def normalize_face(images):

	colored_images = []
	grayscale_images = []

	for instance in images:
		alignedFace = alignFace(instance)
		img, gray_img = detectFace(alignedFace)
		colored_images.append(img)
		grayscale_images.append(gray_img)
		plt.imshow(gray_img, cmap="gray")
		plt.show()
	
	return colored_images, grayscale_images