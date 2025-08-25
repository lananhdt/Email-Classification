import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from .data_loader import load_data

MODEL_NB = "models/naive_bayes.joblib"
MODEL_SVM = "models/tfidf_svm.joblib"

# Train Naive Bayes
def train_naive_bayes(file_path="data/emails.csv"):
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_NB)

    preds = clf.predict(X_test)
    return classification_report(y_test, preds), confusion_matrix(y_test, preds)

def evaluate_naive_bayes(file_path="data/emails.csv"):
    return train_naive_bayes(file_path)

# Train TF-IDF + SVM
def train_svm(file_path="data/emails.csv"):
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("svm", LinearSVC())
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_SVM)

    preds = clf.predict(X_test)
    return classification_report(y_test, preds), confusion_matrix(y_test, preds)

def evaluate_svm(file_path="data/emails.csv"):
    return train_svm(file_path)

# Dự đoán một email
def predict_single(text, model_path=MODEL_NB):
    clf = joblib.load(model_path)
    return clf.predict([text])[0]
