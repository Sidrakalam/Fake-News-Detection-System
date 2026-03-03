from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import re
import nltk
from nltk.corpus import stopwords
import json
import os
from datetime import datetime
import mysql.connector

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CONNECT DATABASE (MOVE THIS UP) ----------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",   # apna MySQL password
    database="fake_news_db"
)

cursor = db.cursor()

# ---------------- STOPWORDS ----------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------- CLEAN TEXT FUNCTION ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'reuters', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return render_template("home.html")

# ---------------- DASHBOARD PAGE ----------------
@app.route('/dashboard')
def dashboard():

    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    # Fetch prediction stats from DB
    cursor.execute("SELECT prediction FROM predictions")
    rows = cursor.fetchall()

    fake_count = 0
    real_count = 0

    for row in rows:
        if "Fake" in row[0]:
            fake_count += 1
        else:
            real_count += 1

    total_predictions = fake_count + real_count

    # Get last 5 predictions
    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5")
    history = cursor.fetchall()

    return render_template(
        "dashboard.html",
        metrics=metrics,
        history=history,
        total_predictions=total_predictions,
        fake_predictions=fake_count,
        real_predictions=real_count
    )

# ---------------- PREDICTION PAGE ----------------
@app.route('/predict-page')
def predict_page():
    return render_template("index.html")

# ---------------- PREDICTION LOGIC ----------------
@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']

    cleaned = clean_text(news_text)
    vect = vectorizer.transform([cleaned])

    prediction = model.predict(vect)[0]
    probability = model.predict_proba(vect)[0]
    confidence = round(max(probability) * 100, 2)

    if prediction == 0:
        result = "Fake News ❌"
        prediction_class = "fake"
    else:
        result = "Real News ✅"
        prediction_class = "real"

    # SAVE TO MYSQL (instead of CSV)
    insert_query = """
    INSERT INTO predictions (news_text, prediction, confidence)
    VALUES (%s, %s, %s)
    """
    cursor.execute(insert_query, (news_text, result, confidence))
    db.commit()

    return render_template(
        "index.html",
        prediction_text=f"{result} (Confidence: {confidence}%)",
        prediction_class=prediction_class
    )

# ---------------- ADMIN LOGIN ----------------
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        query = "SELECT * FROM admins WHERE username=%s AND password=%s"
        cursor.execute(query, (username, password))
        admin = cursor.fetchone()

        if admin:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template("admin_login.html", error="Invalid Credentials")

    return render_template("admin_login.html")

@app.route('/admin-dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    cursor.execute("SELECT prediction FROM predictions")
    rows = cursor.fetchall()

    fake_count = 0
    real_count = 0

    for row in rows:
        if "Fake" in row[0]:
            fake_count += 1
        else:
            real_count += 1

    total = fake_count + real_count

    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    history = cursor.fetchall()

    return render_template(
        "admin_dashboard.html",
        history=history,
        fake_count=fake_count,
        real_count=real_count,
        total=total
    )

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)