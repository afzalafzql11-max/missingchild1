from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sqlite3
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB = "database.db"


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
        photo TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


# -----------------------------
# SIGNUP
# -----------------------------

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

    return jsonify({"message":"Account created"})


# -----------------------------
# LOGIN
# -----------------------------

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
        return jsonify({"status":"success"})
    else:
        return jsonify({"status":"fail"})


# -----------------------------
# REGISTER CHILD
# -----------------------------

@app.route("/register_child", methods=["POST"])
def register_child():

    name = request.form["name"]
    age = request.form["age"]
    place = request.form["place"]
    photo = request.files["photo"]

    path = os.path.join(UPLOAD_FOLDER, photo.filename)
    photo.save(path)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO children(name,age,place,photo) VALUES(?,?,?,?)",
        (name, age, place, path)
    )

    conn.commit()
    conn.close()

    return jsonify({"message":"Child registered successfully"})


# -----------------------------
# CROSS CHECK
# -----------------------------

@app.route("/crosscheck", methods=["POST"])
def crosscheck():

    photo = request.files["photo"]

    upload_path = os.path.join(UPLOAD_FOLDER, photo.filename)
    photo.save(upload_path)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT name,age,place,photo FROM children")
    rows = cur.fetchall()

    conn.close()

    for row in rows:

        db_img = row[3]

        try:

            result = DeepFace.verify(
                img1_path=upload_path,
                img2_path=db_img,
                model_name="VGG-Face",
                enforce_detection=False
            )

            if result["verified"]:

                return jsonify({
                    "status":"found",
                    "name":row[0],
                    "age":row[1],
                    "place":row[2]
                })

        except:
            pass

    return jsonify({"status":"not found"})


# -----------------------------
# HEALTH CHECK (important for Render)
# -----------------------------

@app.route("/")
def home():
    return "Missing Child Detection API Running"


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
