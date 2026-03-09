from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sqlite3
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
DATASET = "dataset"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET, exist_ok=True)

DB = "database.db"


# ---------------- FACE DETECTOR ----------------

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------- DATABASE ----------------

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
        image_path TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ---------------- FACE EXTRACT ----------------

def extract_face(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces)==0:
        return None

    (x,y,w,h)=faces[0]

    face = gray[y:y+h,x:x+w]

    face = cv2.resize(face,(200,200))

    return face


# ---------------- TRAIN MODEL ----------------

def train_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT id,image_path FROM children")
    rows = cur.fetchall()

    conn.close()

    for row in rows:

        img = cv2.imread(row[1],0)

        if img is None:
            continue

        faces.append(img)
        labels.append(row[0])

    if len(faces)==0:
        return None

    recognizer.train(faces,np.array(labels))

    return recognizer


# ---------------- SIGNUP ----------------

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


# ---------------- LOGIN ----------------

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


# ---------------- REGISTER CHILD ----------------

@app.route("/register_child",methods=["POST"])
def register_child():

    name = request.form["name"]
    age = request.form["age"]
    place = request.form["place"]
    photo = request.files["photo"]

    path = os.path.join(DATASET,photo.filename)
    photo.save(path)

    face = extract_face(path)

    if face is None:
        return jsonify({"message":"No face detected"})

    cv2.imwrite(path,face)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
    "INSERT INTO children(name,age,place,image_path) VALUES(?,?,?,?)",
    (name,age,place,path)
    )

    conn.commit()
    conn.close()

    return jsonify({"message":"Child registered"})


# ---------------- CROSSCHECK ----------------

@app.route("/crosscheck",methods=["POST"])
def crosscheck():

    photo = request.files["photo"]

    path = os.path.join(UPLOAD_FOLDER,photo.filename)
    photo.save(path)

    face = extract_face(path)

    if face is None:
        return jsonify({"status":"no face"})

    recognizer = train_model()

    if recognizer is None:
        return jsonify({"status":"database empty"})

    label,confidence = recognizer.predict(face)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT name,age,place FROM children WHERE id=?",(label,))
    row = cur.fetchone()

    conn.close()


    # ---------------- MATCH RULES ----------------

    if confidence < 60:

        return jsonify({
        "status":"found",
        "match_type":"normal",
        "name":row[0],
        "age":row[1],
        "place":row[2]
        })


    elif confidence < 80:

        return jsonify({
        "status":"found",
        "match_type":"age_progression",
        "name":row[0],
        "age":row[1],
        "place":row[2]
        })


    else:

        return jsonify({"status":"not found"})


# ---------------- ROOT ----------------

@app.route("/")
def home():

    return "Missing Child Detection API Running"


# ---------------- RUN ----------------

if __name__=="__main__":

    port = int(os.environ.get("PORT",5000))

    app.run(host="0.0.0.0",port=port)
