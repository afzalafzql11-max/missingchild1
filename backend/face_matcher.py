import face_recognition
import numpy as np
import os
from PIL import Image

DB_PATH = "faces_db"

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)


def register_face(name, image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return False

    encoding = encodings[0]

    np.save(f"{DB_PATH}/{name}.npy", encoding)

    return True


def find_match(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return None

    unknown_encoding = encodings[0]

    for file in os.listdir(DB_PATH):

        known_encoding = np.load(f"{DB_PATH}/{file}")

        result = face_recognition.compare_faces(
            [known_encoding],
            unknown_encoding
        )

        if result[0]:
            return file.replace(".npy", "")

    return None
