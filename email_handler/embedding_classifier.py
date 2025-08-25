import numpy as np
import pandas as pd
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from data_loader import load_data

# Sentence-Transformer (E5 đa ngôn ngữ)
from sentence_transformers import SentenceTransformer

_model = None
_index = None
_train_texts = None
_train_labels = None
_label_set = ["ham","spam"]

def _embed(texts):
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-small")
    # E5: tiền tố "query: " / "passage: " thường dùng khi retriever; ở đây encode trực tiếp
    vecs = _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vecs.astype("float32")

def _prepare():
    global _index, _train_texts, _train_labels
    if _index is not None:
        return
    df = load_data()
    if df.empty:
        df = pd.DataFrame({"text":["hello","win money now"], "label":["ham","spam"]})
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    _prepare.X_test = list(X_test); _prepare.y_test = list(y_test)
    # build FAISS on train embeddings
    emb = _embed(list(X_train))
    dim = emb.shape[1]
    _index = faiss.IndexFlatIP(dim)  # cosine if embeddings normalized → inner product
    _index.add(emb)
    _train_texts = list(X_train)
    _train_labels = list(y_train)

def _majority_vote(nei_indices, k):
    labs = [ _train_labels[i] for i in nei_indices[:k] ]
    # vote: nếu tie → ưu tiên 'spam' nếu xuất hiện (an toàn)
    ham = labs.count("ham"); spam = labs.count("spam")
    if spam >= ham: return "spam"
    return "ham"

def predict_embedding(text: str, k: int = 3) -> str:
    _prepare()
    q = _embed([text])
    D, I = _index.search(q, k=max(k,5))  # lấy dư để tránh đồng hạng; vote theo k
    return _majority_vote(I[0], k)

def _metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="spam", zero_division=0)
    return {"accuracy":acc, "precision":p, "recall":r, "f1":f1}

def evaluate_knn(k_list=[1,3,5]):
    _prepare()
    Xte = _prepare.X_test; yte = _prepare.y_test
    q_emb = _embed(Xte)
    # presearch 5 NN để dùng lại
    D, I = _index.search(q_emb, k=5)

    rows = []
    cms = {}
    for k in k_list:
        preds = []
        for idx in range(len(Xte)):
            pred = _majority_vote(I[idx], k)
            preds.append(pred)
        cm = confusion_matrix(yte, preds, labels=_label_set)
        cms[k] = cm
        m = _metrics(yte, preds)
        rows.append({"k":k, **m})
    metrics_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_k = int(metrics_df.sort_values(["f1","accuracy"], ascending=False).iloc[0]["k"])
    return metrics_df, best_k, cms
