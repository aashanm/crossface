import face_recognition

bill_gates_img = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_gates_encoding = face_recognition.face_encodings(bill_gates_img)[0]

unknown_img = face_recognition.load_image_file('./img/unknown/bill-gates-4.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_img)[0]

# Face Comparison

results = face_recognition.compare_faces([bill_gates_encoding], unknown_encoding)

if (results[0]):
    print('This is Bill Gates')
else:
    print('This is NOT Bill Gates')