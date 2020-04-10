import face_recognition

image = face_recognition.load_image_file('./img/groups/team1.jpg')

face_locations = face_recognition.face_locations(image)

#Array of coordinates of each face

print(face_locations)