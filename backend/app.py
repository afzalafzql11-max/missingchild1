from flask import Flask, request, jsonify
import cv2
import os
import sqlite3
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB = "database.db"


# Face detector
face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def get_face_vector(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:
        return None

    (x,y,w,h) = faces[0]

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face,(100,100))

    return face.flatten()


@app.route("/crosscheck", methods=["POST"])
def crosscheck():

    photo = request.files["photo"]

    path = os.path.join(UPLOAD_FOLDER, photo.filename)
    photo.save(path)

    uploaded_vector = get_face_vector(path)

    if uploaded_vector is None:
        return jsonify({"status":"no face"})


    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT name,age,place,vector FROM children")
    rows = cur.fetchall()

    for row in rows:

        db_vector = np.frombuffer(row[3],dtype=np.uint8)

        distance = np.linalg.norm(uploaded_vector-db_vector)

        if distance < 2000:

            return jsonify({
                "status":"found",
                "name":row[0],
                "age":row[1],
                "place":row[2]
            })

    return jsonify({"status":"not found"})
