# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 18:07:46 2019

@author: Sanjeev
"""
# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import numpy as np


#File direcotry
pathURl="C:/Users/Sanjeev/Desktop/Advance machine learning/Face Detection_Opencv/Face Recog with opencv and deep learning/face-recognition-opencv/face-recognition-opencv"
args = {
	"image": pathURl + '/image'
    }

# load the known faces and embeddings
file = open("encodings.pickle","rb")
data = pickle.load(file)
file.close()
# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"]+"/11.png")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
for encoding in encodings:
	matches = face_recognition.compare_faces(data["encodings"],encoding)
	name = "Unknown"  
	if True in matches:
		
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1
		name = max(counts, key=counts.get)
	# update the list of names
	names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
	cv2.rectangle(image,(left, top),(right,bottom),(0, 255, 0),2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)