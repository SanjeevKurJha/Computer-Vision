# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 18:07:46 2019

@author: Sanjeev
"""
# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

#File direcotry
pathURl="C:/Users/Sanjeev/Desktop/Advance machine learning/Face Detection_Opencv/Face Recog with opencv and deep learning/face-recognition-opencv/face-recognition-opencv"
args = {
	"dataset": pathURl + '/dataset'
    }

# grab the paths to the input images in our dataset
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
value__Encodings = []
face_Names = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("Image process {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb)

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		value__Encodings.append(encoding)
		face_Names.append(name)

#Serializing encodings
data = {"encodings": value__Encodings, "names": face_Names}

file = open("encodings.pickle","wb")
pickle.dump(data, file)
file.close()



"""
file = open("encodings.pickle","rb")
data = pickle.load(file)

# close the file
file.close()

data['encodings'][0]
data['names']

#file = open(args["encodings"], "wb")
#pickle.dumps(data,file)
#file.close()

"""