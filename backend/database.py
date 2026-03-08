import sqlite3

def init_db():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS missing(
        id INTEGER PRIMARY KEY,
        name TEXT,
        location TEXT,
        email TEXT,
        image TEXT
    )
    """)

    conn.commit()
    conn.close()