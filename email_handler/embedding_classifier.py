import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from .data_loader import load_data

MODEL_EMB = "models/embed_model.joblib"
MODEL_KNN = "models/knn_model.joblib"

# Load Sentence-BERT
def load_embed_model(name="intfloat/multilingual-e5-base"):
    try:
        return joblib.load(MODEL_EMB)
    except:
        model = SentenceTransformer(name)
        joblib.dump(model, MODEL_EMB)
        return model

# Tạo embedding cho văn bản
def embed_texts(texts, model):
    return np.array(model.encode(texts, normalize_embeddings=True))

# Train KNN + Embedding
def train_knn(file_path="data/emails.csv"):
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    model = load_embed_model()
    X_train_emb = embed_texts(X_train.tolist(), model)
    X_test_emb = embed_texts(X_test.tolist(), model)

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X_train_emb, y_train)
    joblib.dump(knn, MODEL_KNN)

    # ✅ tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)

    joblib.dump(clf, MODEL_KNN)
    joblib.dump(embedder, MODEL_EMBEDDING)

    print(f"✅ KNN model saved at {MODEL_KNN}")
    print(f"✅ Embedding model saved at {MODEL_EMBEDDING}")
    
    preds = knn.predict(X_test_emb)
    return classification_report(y_test, preds), confusion_matrix(y_test, preds)

def evaluate_knn(file_path="data/emails.csv"):
    return train_knn(file_path)
