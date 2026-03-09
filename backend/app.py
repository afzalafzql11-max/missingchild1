from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sqlite3
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB = "database.db"


# -----------------------------
# FACE DETECTOR
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# -----------------------------
# DATABASE INIT
# -----------------------------

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
        age INTEGER,
        place TEXT,
        vector BLOB
    )
    """)

    conn.commit()
    conn.close()

init_db()


# -----------------------------
# FACE VECTOR GENERATOR
# -----------------------------

def get_face_vector(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:
        return None

    (x,y,w,h) = faces[0]

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face,(120,120))

    face = cv2.equalizeHist(face)

    return face.flatten()


# -----------------------------
# REVERSE AGE PROGRESSION
# -----------------------------

def reverse_age_progression(vector):

    face = vector.reshape(120,120).astype("uint8")

    versions = []

    # smoother skin
    smooth = cv2.GaussianBlur(face,(7,7),0)
    versions.append(smooth.flatten())

    # brightness change
    bright = cv2.convertScaleAbs(face,alpha=1.2,beta=20)
    versions.append(bright.flatten())

    # histogram equalization
    hist = cv2.equalizeHist(face)
    versions.append(hist.flatten())

    # sharpening
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(face,-1,kernel)
    versions.append(sharp.flatten())

    return versions


# -----------------------------
# SIGNUP
# -----------------------------

@app.route("/signup",methods=["POST"])
def signup():

    data = request.json

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
    "INSERT INTO users(name,email,password) VALUES(?,?,?)",
    (data["name"],data["email"],data["password"])
    )

    conn.commit()
    conn.close()

    return jsonify({"message":"Account created"})


# -----------------------------
# LOGIN
# -----------------------------

@app.route("/login",methods=["POST"])
def login():

    data = request.json

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
    "SELECT * FROM users WHERE email=? AND password=?",
    (data["email"],data["password"])
    )

    user = cur.fetchone()

    conn.close()

    if user:
        return jsonify({"status":"success"})
    else:
        return jsonify({"status":"fail"})


# -----------------------------
# REGISTER CHILD
# -----------------------------

@app.route("/register_child",methods=["POST"])
def register_child():

    name = request.form["name"]
    age = request.form["age"]
    place = request.form["place"]
    photo = request.files["photo"]

    path = os.path.join(UPLOAD_FOLDER,photo.filename)
    photo.save(path)

    vector = get_face_vector(path)

    if vector is None:
        return jsonify({"message":"No face detected"})

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
    "INSERT INTO children(name,age,place,vector) VALUES(?,?,?,?)",
    (name,age,place,vector.tobytes())
    )

    conn.commit()
    conn.close()

    return jsonify({"message":"Child registered"})


# -----------------------------
# CROSS CHECK
# -----------------------------

@app.route("/crosscheck",methods=["POST"])
def crosscheck():

    photo = request.files["photo"]

    path = os.path.join(UPLOAD_FOLDER,photo.filename)
    photo.save(path)

    uploaded_vector = get_face_vector(path)

    if uploaded_vector is None:
        return jsonify({"status":"no face"})

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT name,age,place,vector FROM children")

    rows = cur.fetchall()

    conn.close()


    # ---------------- NORMAL MATCH ----------------

    for row in rows:

        db_vector = np.frombuffer(row[3],dtype=np.uint8)

        distance = np.linalg.norm(uploaded_vector-db_vector)

        if distance < 2000:

            return jsonify({
            "status":"found",
            "match_type":"normal",
            "name":row[0],
            "age":row[1],
            "place":row[2]
            })


    # ---------------- AGE TOLERANCE MATCH ----------------

    for row in rows:

        db_vector = np.frombuffer(row[3],dtype=np.uint8)

        distance = np.linalg.norm(uploaded_vector-db_vector)

        if distance < 3500:

            return jsonify({
            "status":"found",
            "match_type":"age_progression",
            "name":row[0],
            "age":row[1],
            "place":row[2]
            })


    # ---------------- REVERSE AGE PROGRESSION ----------------

    aged_versions = reverse_age_progression(uploaded_vector)

    for aged_vector in aged_versions:

        for row in rows:

            db_vector = np.frombuffer(row[3],dtype=np.uint8)

            distance = np.linalg.norm(aged_vector-db_vector)

            if distance < 4000:

                return jsonify({
                "status":"found",
                "match_type":"reverse_age_progression",
                "name":row[0],
                "age":row[1],
                "place":row[2]
                })


    # ---------------- NOT FOUND ----------------

    return jsonify({"status":"not found"})


# -----------------------------
# ROOT (RENDER HEALTH CHECK)
# -----------------------------

@app.route("/")
def home():
    return "Missing Child Detection API Running"


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",5000))

    app.run(host="0.0.0.0",port=port)
