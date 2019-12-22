# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:55:00 2018

@author: Sanjeev Jha
"""
#Import OpenCV

import cv2
import matplotlib.pyplot as plt
import time

#def convertToRGB(img): 
#    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#load test iamge
test1 = cv2.imread('Group.jpg')
#convert the test image to gray image as opencv face detector expects gray images 
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

#if you have matplotlib installed then  
#plt.imshow(gray_img, cmap='gray')  

# or display the gray image using OpenCV 
cv2.imshow('Test Imag', gray_img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#let's detect multiscale (some images may be closer to camera than others) images 
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 

#print the number of faces found 
print ("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", test1)
cv2.waitKey(0)

"""
def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.         
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)          
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              
    return img_copy

 
faces_detected_img = detect_faces(haar_face_cascade, test1)
#convert image to RGB and show image 
plt.imshow(convertToRGB(faces_detected_img)) 

"""


