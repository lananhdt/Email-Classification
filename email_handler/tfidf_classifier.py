import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from .data_loader import load_data

# Cache đơn giản ở module scope
_vectorizer = None
_model_svm = None
_model_nb = None
_labels = ["ham", "spam"]

def _prepare():
    global _vectorizer, _model_svm, _model_nb
    if _vectorizer is not None and _model_svm is not None and _model_nb is not None:
        return
    df = load_data()
    if df.empty:  # tránh vỡ
        df = pd.DataFrame({"text":["hello","win money now"], "label":["ham","spam"]})
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    _vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    Xtr = _vectorizer.fit_transform(X_train)
    Xte = _vectorizer.transform(X_test)
    # models
    _model_svm = LinearSVC()
    _model_svm.fit(Xtr, y_train)
    _model_nb = MultinomialNB()
    _model_nb.fit(Xtr, y_train)
    # store test for evaluation
    _prepare.Xte, _prepare.yte = Xte, y_test

def _metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="spam", zero_division=0)
    return {"accuracy":acc, "precision":p, "recall":r, "f1":f1}

def predict_tfidf(text: str) -> str:
    _prepare()
    X = _vectorizer.transform([text])
    pred = _model_svm.predict(X)[0]
    return pred

def evaluate_svm(return_metrics=False):
    _prepare()
    y_pred = _model_svm.predict(_prepare.Xte)
    rep = classification_report(_prepare.yte, y_pred, digits=4)
    cm = confusion_matrix(_prepare.yte, y_pred, labels=_labels)
    if return_metrics:
        return rep, cm, _metrics(_prepare.yte, y_pred)
    return rep, cm

def evaluate_naive_bayes(return_metrics=False):
    _prepare()
    y_pred = _model_nb.predict(_prepare.Xte)
    rep = classification_report(_prepare.yte, y_pred, digits=4)
    cm = confusion_matrix(_prepare.yte, y_pred, labels=_labels)
    if return_metrics:
        return rep, cm, _metrics(_prepare.yte, y_pred)
    return rep, cm
