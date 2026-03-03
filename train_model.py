import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake['label'] = 0
true['label'] = 1

data = pd.concat([fake, true], axis=0)

data = data[['text', 'label']]
data = data.dropna()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class Distribution:")
print(data['label'].value_counts())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'reuters', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

print("Cleaning text...")
data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['label']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(max_df=0.7, ngram_range=(1,2))

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")


import json

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "total_data": int(len(data)),
    "fake_count": int(sum(data['label'] == 0)),
    "real_count": int(sum(data['label'] == 1)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)