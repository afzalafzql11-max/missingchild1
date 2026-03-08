from flask import Flask, request, jsonify
import os
import sqlite3

from face_matcher import register_face, find_match  # <- updated code
from enhance_gan import enhance_image  # you can keep this
from age_gan import age_progression   # keep if you want simulated age progression
from database import init_db

app = Flask(__name__)

# Create required folders
os.makedirs("uploads/missing", exist_ok=True)
os.makedirs("uploads/found", exist_ok=True)
os.makedirs("faces_db", exist_ok=True)

# Initialize database
init_db()


@app.route("/register_missing", methods=["POST"])
def register_missing():
    name = request.form["name"]
    location = request.form["location"]
    email = request.form["email"]
    image = request.files["image"]

    path = os.path.join("uploads/missing", image.filename)
    image.save(path)

    # Register face in the face_recognition DB
    register_face(name, path)

    # Save metadata in SQLite
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO missing(name,location,email,image) VALUES (?,?,?,?)",
        (name, location, email, path)
    )
    conn.commit()
    conn.close()

    return jsonify({"message": "Child registered"})


@app.route("/find_child", methods=["POST"])
def find_child():
    image = request.files["image"]
    path = os.path.join("uploads/found", image.filename)
    image.save(path)

    # Check for direct face match
    match = find_match(path)

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    if match:
        # Delete from DB and return match
        c.execute("DELETE FROM missing WHERE name=?", (match,))
        conn.commit()
        conn.close()
        return jsonify({"result": "MATCH FOUND", "name": match})

    # If no match → enhance / age progression
    enhanced_path = enhance_image(path)  # optional, if you have enhancement
    aged_path = age_progression(enhanced_path or path)

    match_after_age = find_match(aged_path)
    if match_after_age:
        c.execute("DELETE FROM missing WHERE name=?", (match_after_age,))
        conn.commit()
        conn.close()
        return jsonify({"result": "MATCH AFTER AGE PROGRESSION", "name": match_after_age})

    conn.close()
    return jsonify({"result": "NOT FOUND"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
