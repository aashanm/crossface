import face_recognition
import os
import cv2 
import numpy as numpy
from PIL import Image
import pickle
import time

#directories
known_faces_dir = './img/known'

#imaging parameters
tolerance = 0.6
frame_thickness = 3
font_thickness = 2
model_name = "cnn" #deep-learning pre-trained model

video = cv2.VideoCapture("insertvideohere.mp4")

print("Loading known faces...")

#encoded known faces
known_faces = []
known_names = []

#add known names and encodings to known faces lists
for name in os.listdir(known_faces_dir):
    #load all files of known faces/images
    for filename in os.listdir(f"{known_faces_dir}/{name}"):
        #load image
        encoding = face_recognition.face_encodings

        #get facial encoding characters, take only the first face encoding
        encoding = pickle.load(open(f'{name}/{filename}', 'rb'))

        #add names and encodings to lists
        known_faces.append(encoding)
        known_names.append(name)

if len(known_names) > 0:
    next_id = max(known_names) + 1
else: 
    next_id = 0

print("Processing unknown faces...")

#loop through unknown faces directory that we want to label
while True:
    
    ret, image = video.read()

    #retreive locations to draw boxes around images
    locs = face_recognition.face_locations(image, model = model_name)
    encodings = face_recognition.face_encodings(image, locs)

    #iterate through unknown images and check for matches (assuming possibility of multiple faces in image)
    for face_encoding, face_location in zip(encodings, locs):
        
        #returns array of true/false values for each image
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        print (results)

        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match Found: {match}")
        else:
            match = str(next_id)
            next_id+=1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f'{known_faces_dir}/{match}')
            pickle.dump(face_encoding, open(f'{known_faces_dir}/{match}/{match}-{int(time.time())}.pkl', "wb"})

        #Set framing coordinates
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        color = [0, 255, 0]

        cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)

        #Set labelling coordinates
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font_thickness)

    #Display images
    cv2.imshow("", image)
    if cv2.waitkey(1) & 0xFF == ord("q"):
        break

