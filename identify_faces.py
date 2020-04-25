import face_recognition
from PIL import Image, ImageDraw

bill_gates_img = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_gates_encoding = face_recognition.face_encodings(bill_gates_img)[0]

steve_jobs_img = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_jobs_encoding = face_recognition.face_encodings(steve_jobs_img)[0]

known_encodings = [
    bill_gates_encoding,
    steve_jobs_encoding
]

known_names = [
    "Bill Gates",
    "Steve Jobs"
]

test_img = face_recognition.load_image_file('./img/groups/bill-steve.jpg')

face_locs = face_recognition.face_locations(test_img)
face_encodings = face_recognition.face_encodings(test_img, face_locs)

pil_img = Image.fromarray(test_img)

draw = ImageDraw.Draw(pil_img)

for(top, right, bottom, left), face_encoding in zip(face_locs, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)

    name = "Unknown"

    if True in matches:
        match_index1 = matches.index(True)
        name = known_names[match_index1]
    
    draw.rectangle(((left, top), (right, bottom)), outline = (255, 0, 0))

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill = (255, 0, 0), outline = (255, 0, 0))
    draw.text((left + 6, bottom - text_height - 5), name, fill = (255, 255, 255, 255))

del draw

pil_img.show()

