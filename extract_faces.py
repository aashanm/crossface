from PIL import Image
import face_recognition

img = face_recognition.load_image_file('./img/groups/team1.jpg')
face_locs = face_recognition.face_locations(img)

for face_loc in face_locs:
    top, right, bottom, left = face_loc

    face_img = img[top:bottom, left:right]
    pil_img = Image.fromarray(face_img)
    pil_img.show()