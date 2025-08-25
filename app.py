import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ====== Imports từ package email_handler ======
from email_handler.data_loader import load_data
from email_handler.corrections_handler import (
    count_corrections, save_correction, get_corrections_df
)
from email_handler.tfidf_classifier import (
    predict_tfidf, evaluate_svm, evaluate_naive_bayes
)
from email_handler.embedding_classifier import (
    predict_embedding, evaluate_knn
)
from email_handler.gmail_handler import (
    get_gmail_service,
    get_email_list_with_ids,  # có id + snippet
    get_email_list,           # fallback: chỉ snippet
    ensure_label, add_label, move_message
)

# ================== PAGE CONFIG & THEME ==================
if 'page' not in st.session_state:
    st.session_state.page = "Tổng quan"
    
st.set_page_config(page_title="Email Classifier", layout="wide", page_icon="📧")

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Title */
.big-title {
  font-size: 44px; font-weight: 900;
  background: linear-gradient(90deg,#a7f3d0,#22c55e);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0 0 6px 0;
}
.subtitle { color:#cbd5e1; font-size:16px; margin-bottom: 28px; }

/* Cards */
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px; padding: 16px;
}

/* Buttons */
.stButton>button { border-radius: 16px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ================== DATA ==================
DATA_PATH = "data/emails.csv"
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.warning(f"Không thể load dữ liệu từ {DATA_PATH}: {e}")
    df = pd.DataFrame(columns=["text","label"])

# ================== SIDEBAR NAV ==================
st.sidebar.title("📌 MENU")
if "page" not in st.session_state:
    st.session_state.page = "🏠 Trang chủ"

PAGES = [
    "🏠 Trang chủ",
    "📊 Phân tích dữ liệu & Thống kê",
    "🧪 Đánh giá mô hình",
    "📧 Quét Gmail & Corrections"
]
choice = st.sidebar.radio("Chọn chức năng", PAGES, index=PAGES.index(st.session_state.page))
st.session_state.page = choice

# ================== HELPERS ==================
def draw_cm(cm, labels=("ham","spam"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig)

def goto(page_name: str):
    st.session_state.page = page_name
    st.experimental_rerun()

def get_items_safe(service, max_emails=10):
    """Cố gắng lấy {id, snippet}; nếu không được thì fallback chỉ snippet."""
    try:
        items = get_email_list_with_ids(service, max_results=max_emails)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return items
    except Exception:
        pass
    try:
        snippets = get_email_list(service, max_results=max_emails)
        return [{"id": None, "snippet": s} for s in snippets]
    except Exception:
        return []

# ================== PAGES ==================
def page_home():
    st.markdown('<div class="big-title">Email Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Khám phá và phân loại email với giao diện tương tác!</div>', unsafe_allow_html=True)

    total = len(df)
    spam_count = int((df["label"]=="spam").sum()) if "label" in df else 0
    ham_count  = int((df["label"]=="ham").sum()) if "label" in df else 0
    try:
        corrections = count_corrections()
    except Exception:
        corrections = 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tổng số Email", f"{total:,}")
    c2.metric("Email Spam", f"{spam_count:,}")
    c3.metric("Email Ham", f"{ham_count:,}")
    c4.metric("Corrections", f"{corrections:,}")

    st.markdown("#### Tính năng")
    col1, col2, col3 = st.columns(3)
    if col1.button("📊 Phân tích & Thống kê", use_container_width=True):
        goto("📊 Phân tích dữ liệu & Thống kê")
    if col2.button("🧪 Đánh giá Mô hình", use_container_width=True):
        goto("🧪 Đánh giá mô hình")
    if col3.button("📧 Quét Gmail", use_container_width=True):
        goto("📧 Quét Gmail & Corrections")

def page_analysis_overview():
    st.header("📊 Phân tích dữ liệu & Thống kê")

    if df.empty:
        st.info("Dataset trống.")
        return

    # --- 1) Tổng quan Dataset ---
    st.subheader("1) Tổng quan Dataset (5 dòng đầu)")
    st.dataframe(df.head())

    # --- 2) Phân phối Spam/Ham ---
    st.subheader("2) Phân phối Spam vs Ham")
    fig, ax = plt.subplots()
    sns.countplot(x="label", data=df, ax=ax)
    ax.set_xlabel("Label"); ax.set_ylabel("Count")
    st.pyplot(fig)

    # --- 3) t-SNE Visualization ---
    st.subheader("3) Minh hoạ t-SNE (TF-IDF 200 features, tối đa 1000 mẫu)")
    sample = df.sample(min(1000, len(df)), random_state=42) if len(df) > 0 else df
    if not sample.empty:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(max_features=200).fit_transform(sample["text"])
        X_emb = TSNE(n_components=2, random_state=42, init="random", perplexity=30).fit_transform(X.toarray())
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=sample["label"], s=18, ax=ax, palette="Set1")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(loc="best")
        st.pyplot(fig)
    else:
        st.info("Không đủ dữ liệu để vẽ t-SNE.")

def page_evaluate_models():
    st.header("🧪 Đánh giá mô hình")

    # ---- TF-IDF + SVM ----
    st.subheader("TF-IDF + SVM (LinearSVC) — baseline")
    svm_metrics = None
    try:
        # ưu tiên API có return_metrics=True
        try:
            report_svm, cm_svm, svm_metrics = evaluate_svm(return_metrics=True)
        except TypeError:
            report_svm, cm_svm = evaluate_svm()
        st.text(report_svm)
        draw_cm(cm_svm, title="TF-IDF + SVM")
    except Exception as e:
        st.error(f"Lỗi SVM: {e}")

    # ---- Naive Bayes ----
    st.subheader("Naive Bayes (MultinomialNB)")
    nb_metrics = None
    try:
        try:
            report_nb, cm_nb, nb_metrics = evaluate_naive_bayes(return_metrics=True)
        except TypeError:
            report_nb, cm_nb = evaluate_naive_bayes()
        st.text(report_nb)
        draw_cm(cm_nb, title="Naive Bayes")
    except Exception as e:
        st.error(f"Lỗi NB: {e}")

    # ---- KNN + FAISS + E5 ----
    st.subheader("KNN + FAISS + E5 (k ∈ {1,3,5})")
    metrics_df = None; best_k = None; cms = {}
    try:
        metrics_df, best_k, cms = evaluate_knn(k_list=[1,3,5])
        st.dataframe(metrics_df)

        # Lineplot theo k
        fig, ax = plt.subplots()
        m = metrics_df.melt(id_vars=["k"], var_name="metric", value_name="score")
        sns.lineplot(data=m, x="k", y="score", hue="metric", marker="o", ax=ax)
        ax.set_ylim(0,1); ax.set_title("KNN metrics theo k")
        st.pyplot(fig)

        # CM theo k
        for k, cm in cms.items():
            draw_cm(cm, title=f"KNN (k={k})")
    except Exception as e:
        st.error(f"Lỗi KNN: {e}")

    # ---- So sánh TF-IDF vs KNN best-k ----
    st.subheader("So sánh TF-IDF vs KNN (best-k)")
    try:
        if metrics_df is not None and best_k is not None and svm_metrics is not None:
            knn_best = metrics_df[metrics_df["k"]==best_k].iloc[0]
            comp = pd.DataFrame([
                {"model":"TF-IDF+SVM",
                 "Accuracy":svm_metrics.get("accuracy", np.nan),
                 "Precision":svm_metrics.get("precision", np.nan),
                 "Recall":svm_metrics.get("recall", np.nan),
                 "F1":svm_metrics.get("f1", np.nan)},
                {"model":f"KNN (k={best_k})",
                 "Accuracy":knn_best["accuracy"],
                 "Precision":knn_best["precision"],
                 "Recall":knn_best["recall"],
                 "F1":knn_best["f1"]}
            ])
            st.dataframe(comp)
            fig, ax = plt.subplots()
            cmelt = comp.melt(id_vars=["model"], var_name="metric", value_name="score")
            sns.barplot(data=cmelt, x="metric", y="score", hue="model", ax=ax)
            ax.set_ylim(0,1); ax.set_title("TF-IDF vs KNN best-k")
            st.pyplot(fig)
        else:
            st.info("Chưa đủ metrics để so sánh.")
    except Exception as e:
        st.info(f"Không thể vẽ so sánh: {e}")

def page_gmail_and_corrections():
    st.header("📧 Quét Gmail & Corrections")
    st.markdown("Luồng: **OAuth → Quét email thật → Phân loại → Gán nhãn SPAM/INBOX → Sửa nhãn → Lưu `corrections.json`**")

    colA,colB = st.columns([1,1])
    model_choice = colA.selectbox("Chọn mô hình", ["TF-IDF + SVM", "KNN + E5 Embedding"])
    max_emails = colB.slider("Số email cần quét", 5, 50, 10)

    if st.button("🔐 Kết nối Gmail & Quét"):
        try:
            service = get_gmail_service()
            items = get_items_safe(service, max_emails=max_emails)
            st.success(f"Đã lấy {len(items)} email")

            for i, it in enumerate(items, 1):
                msg_id = it.get("id")
                text = it.get("snippet","")

                with st.expander(f"Email {i} — id: {msg_id}"):
                    st.write(text)

                    # Phân loại
                    try:
                        if model_choice.startswith("TF"):
                            pred = predict_tfidf(text)
                        else:
                            pred = predict_embedding(text)
                    except Exception as e:
                        pred = f"error: {e}"
                    st.write(f"**Kết quả:** `{pred}`")

                    # Hành động trên Gmail
                    c1,c2,c3 = st.columns(3)
                    if c1.button("🏷️ Gán label AI_CORRECTED", key=f"lab_{msg_id}"):
                        try:
                            lid = ensure_label(service, "AI_CORRECTED")
                            if msg_id:
                                add_label(service, msg_id, lid)
                            st.success("Đã gán label.")
                        except Exception as e:
                            st.warning(f"Không gán được label: {e}")

                    if c2.button("🗂️ Chuyển INBOX", key=f"inb_{msg_id}"):
                        try:
                            if msg_id:
                                move_message(service, msg_id, to_spam=False)
                            st.success("Đã chuyển INBOX.")
                        except Exception as e:
                            st.warning(f"Lỗi move INBOX: {e}")

                    if c3.button("🧹 Chuyển SPAM", key=f"spm_{msg_id}"):
                        try:
                            if msg_id:
                                move_message(service, msg_id, to_spam=True)
                            st.success("Đã chuyển SPAM.")
                        except Exception as e:
                            st.warning(f"Lỗi move SPAM: {e}")

                    # Correction thủ công
                    st.markdown("**Sửa nhãn (Correction):**")
                    new_label = st.radio("Chọn nhãn đúng", ["spam","ham"], key=f"corr_{msg_id}", horizontal=True)
                    if st.button("💾 Lưu Correction", key=f"save_{msg_id}"):
                        try:
                            save_correction(text, new_label)
                            st.success("Đã lưu correction.")
                        except Exception as e:
                            st.error(f"Lỗi lưu correction: {e}")

        except Exception as e:
            st.error(f"Gmail error: {e}")
            st.info("Hãy đặt `credentials.json` tại thư mục gốc, và bật Gmail API trong Google Cloud Console.")

    st.markdown("---")
    st.subheader("📋 Corrections đã lưu")
    try:
        st.dataframe(get_corrections_df())
    except Exception as e:
        st.info(f"Chưa có corrections: {e}")

# ================== ROUTER ==================
if st.session_state.page == "🏠 Trang chủ":
    page_home()
elif st.session_state.page == "📊 Phân tích dữ liệu & Thống kê":
    page_analysis_overview()
elif st.session_state.page == "🧪 Đánh giá mô hình":
    page_evaluate_models()
elif st.session_state.page == "📧 Quét Gmail & Corrections":
    page_gmail_and_corrections()
