from flask import Flask, request, jsonify
import sqlite3
import os
import face_recognition
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB = "database.db"


# ---------------- DATABASE ---------------- #

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS children(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age TEXT,
        place TEXT,
        photo TEXT,
        encoding BLOB
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ---------------- AGE PROGRESSION ---------------- #

def age_progression(image_path):

    img = Image.open(image_path)

    # simulate aging
    img = img.filter(ImageFilter.DETAIL)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    aged_path = image_path.replace(".jpg", "_aged.jpg")
    img.save(aged_path)

    return aged_path


# ---------------- SIGNUP ---------------- #

@app.route("/signup", methods=["POST"])
def signup():

    data = request.json

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO users(name,email,password) VALUES(?,?,?)",
        (data["name"], data["email"], data["password"])
    )

    conn.commit()
    conn.close()

    return jsonify({"message": "Account Created"})


# ---------------- LOGIN ---------------- #

@app.route("/login", methods=["POST"])
def login():

    data = request.json

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (data["email"], data["password"])
    )

    user = cur.fetchone()

    conn.close()

    if user:
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "fail"})


# ---------------- REGISTER CHILD ---------------- #

@app.route("/register_child", methods=["POST"])
def register_child():

    name = request.form["name"]
    age = request.form["age"]
    place = request.form["place"]

    photo = request.files["photo"]

    path = os.path.join(UPLOAD_FOLDER, photo.filename)
    photo.save(path)

    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"message": "No face detected"})

    encoding = encodings[0].tobytes()

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO children(name,age,place,photo,encoding) VALUES(?,?,?,?,?)",
        (name, age, place, path, encoding)
    )

    conn.commit()
    conn.close()

    return jsonify({"message": "Child Registered"})


# ---------------- FACE MATCH FUNCTION ---------------- #

def match_face(uploaded_encoding):

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT name,age,place,encoding FROM children")
    rows = cur.fetchall()

    for row in rows:

        db_encoding = np.frombuffer(row[3], dtype=np.float64)

        match = face_recognition.compare_faces(
            [db_encoding], uploaded_encoding
        )[0]

        if match:
            conn.close()
            return {
                "status": "found",
                "name": row[0],
                "age": row[1],
                "place": row[2]
            }

    conn.close()

    return None


# ---------------- CROSS CHECK ---------------- #

@app.route("/crosscheck", methods=["POST"])
def crosscheck():

    photo = request.files["photo"]

    path = os.path.join(UPLOAD_FOLDER, photo.filename)
    photo.save(path)

    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"status": "no face"})

    uploaded_encoding = encodings[0]

    # First match attempt
    result = match_face(uploaded_encoding)

    if result:
        return jsonify(result)

    # Age progression if no match
    aged_path = age_progression(path)

    aged_image = face_recognition.load_image_file(aged_path)
    aged_encodings = face_recognition.face_encodings(aged_image)

    if len(aged_encodings) == 0:
        return jsonify({"status": "not found"})

    aged_encoding = aged_encodings[0]

    result = match_face(aged_encoding)

    if result:
        return jsonify(result)

    return jsonify({"status": "not found"})


# ---------------- RUN SERVER ---------------- #

if __name__ == "__main__":
    app.run()
