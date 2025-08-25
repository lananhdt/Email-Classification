import os
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


# ===================== TRAIN MODELS =====================
def train_naive_bayes(file_path="data/emails.csv"):
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_NB)

    print(f"✅ Naive Bayes model saved at {MODEL_NB}")

    preds = clf.predict(X_test)
    return classification_report(y_test, preds), confusion_matrix(y_test, preds)


def train_svm(file_path="data/emails.csv"):
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("svm", LinearSVC())
    ])
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_SVM)

    print(f"✅ SVM model saved at {MODEL_SVM}")

    preds = clf.predict(X_test)
    return classification_report(y_test, preds), confusion_matrix(y_test, preds)


# ===================== EVALUATE MODELS =====================
def evaluate_naive_bayes(file_path="data/emails.csv"):
    if os.path.exists(MODEL_NB):
        print("ℹ️ Loading existing Naive Bayes model...")
        clf = joblib.load(MODEL_NB)

        df = load_data(file_path)
        _, X_test, _, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )

        preds = clf.predict(X_test)
        return classification_report(y_test, preds), confusion_matrix(y_test, preds)
    else:
        return train_naive_bayes(file_path)


def evaluate_svm(file_path="data/emails.csv"):
    if os.path.exists(MODEL_SVM):
        print("ℹ️ Loading existing SVM model...")
        clf = joblib.load(MODEL_SVM)

        df = load_data(file_path)
        _, X_test, _, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )

        preds = clf.predict(X_test)
        return classification_report(y_test, preds), confusion_matrix(y_test, preds)
    else:
        return train_svm(file_path)


# ===================== PREDICT SINGLE =====================
def predict_single(text, model_path=MODEL_NB):
    clf = joblib.load(model_path)
    return clf.predict([text])[0]
