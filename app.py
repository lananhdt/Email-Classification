import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from data_loader import load_data
from corrections_handler import count_corrections, save_correction, get_corrections_df

from tfidf_classifier import (
    predict_tfidf,
    evaluate_svm,
    evaluate_naive_bayes,
)
from embedding_classifier import (
    predict_embedding,
    evaluate_knn,   # must return (metrics_df, best_k, cms_by_k)
)

from gmail_handler import (
    get_gmail_service,
    get_email_list_with_ids,
    get_email_list,           # back-compat (snippets only)
    ensure_label, add_label,
    move_message,
)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Email Classifier", layout="wide", page_icon="📧")

# ---------- THEME / CSS ----------
st.markdown("""
<style>
/* container */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
/* title */
.big-title {
  font-size: 44px; font-weight: 900;
  background: linear-gradient(90deg,#a7f3d0,#22c55e);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin: 0 0 6px 0;
}
.subtitle { color:#cbd5e1; font-size:16px; margin-bottom: 28px; }
/* cards */
.card { background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
  border-radius:18px; padding:16px; }
/* buttons */
.stButton>button { border-radius:16px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
DATA_PATH = "data/emails.csv"
df = load_data(DATA_PATH)

# ---------- SIDEBAR ----------
st.sidebar.title("📌 MENU")
page = st.sidebar.radio(
    "Chọn chức năng",
    ["🏠 Trang chủ",
     "📊 Tổng quan & Thống kê",
     "🔍 Phân tích dữ liệu & t-SNE",
     "🧪 Đánh giá mô hình",
     "📧 Quét Gmail & Corrections"]
)

# ---------- HELPERS ----------
def draw_cm(cm, labels=("ham","spam"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig)

# ---------- PAGES ----------
def page_home():
    st.markdown('<div class="big-title">Email Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Khám phá và phân loại email với giao diện tương tác!</div>', unsafe_allow_html=True)

    total = len(df)
    spam_count = int((df["label"]=="spam").sum()) if "label" in df else 0
    ham_count  = int((df["label"]=="ham").sum()) if "label" in df else 0
    corrections = count_corrections()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tổng số Email", f"{total:,}")
    c2.metric("Email Spam", f"{spam_count:,}", "↑")
    c3.metric("Email Ham", f"{ham_count:,}", "↑")
    c4.metric("Corrections", f"{corrections:,}")

    st.markdown("#### Tính năng")
    a,b,c,d = st.columns(4)
    a.button("🔬 Phân tích Dữ liệu", on_click=lambda: st.session_state.update(page="🔍 Phân tích dữ liệu & t-SNE"))
    b.button("🧪 Đánh giá Bộ phân loại", on_click=lambda: st.session_state.update(page="🧪 Đánh giá mô hình"))
    c.button("📧 Quét Gmail", on_click=lambda: st.session_state.update(page="📧 Quét Gmail & Corrections"))
    d.button("📝 Quản lý Corrections", on_click=lambda: st.session_state.update(page="📧 Quét Gmail & Corrections"))

def page_overview():
    st.header("📊 Tổng quan & Thống kê")
    if df.empty:
        st.info("Dataset trống.")
        return
    st.subheader("1) Mẫu dữ liệu")
    st.dataframe(df.head())
    st.subheader("2) Phân phối Spam/Ham")
    fig, ax = plt.subplots()
    sns.countplot(x="label", data=df, ax=ax)
    st.pyplot(fig)

def page_analysis_tsne():
    st.header("🔍 Phân tích dữ liệu & t-SNE")
    if df.empty:
        st.info("Dataset trống.")
        return

    st.subheader("t-SNE trên TF-IDF (200 features, sample ≤ 1000)")
    sample = df.sample(min(1000, len(df)), random_state=42)
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(max_features=200).fit_transform(sample["text"])
    X_emb = TSNE(n_components=2, random_state=42, init="random", perplexity=30).fit_transform(X.toarray())
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=sample["label"], s=18, ax=ax, palette="Set1")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    st.pyplot(fig)

def page_eval():
    st.header("🧪 Đánh giá mô hình")

    st.subheader("TF-IDF + SVM (LinearSVC) – baseline")
    try:
        report_svm, cm_svm, svm_metrics = evaluate_svm(return_metrics=True)
        st.text(report_svm); draw_cm(cm_svm, title="TF-IDF + SVM")
    except Exception as e:
        st.error(f"Lỗi SVM: {e}"); svm_metrics = None

    st.subheader("Naive Bayes (MultinomialNB)")
    try:
        report_nb, cm_nb, nb_metrics = evaluate_naive_bayes(return_metrics=True)
        st.text(report_nb); draw_cm(cm_nb, title="Naive Bayes")
    except Exception as e:
        st.error(f"Lỗi NB: {e}"); nb_metrics = None

    st.subheader("KNN + FAISS + E5 (k ∈ {1,3,5})")
    try:
        metrics_df, best_k, cms = evaluate_knn(k_list=[1,3,5])
        st.dataframe(metrics_df)
        # Lineplot k
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

    st.subheader("So sánh TF-IDF vs KNN (best-k)")
    try:
        knn_best = metrics_df[metrics_df["k"]==best_k].iloc[0]
        comp = pd.DataFrame([
            {"model":"TF-IDF+SVM", "Accuracy":svm_metrics["accuracy"], "Precision":svm_metrics["precision"],
             "Recall":svm_metrics["recall"], "F1":svm_metrics["f1"]},
            {"model":f"KNN (k={best_k})", "Accuracy":knn_best["accuracy"], "Precision":knn_best["precision"],
             "Recall":knn_best["recall"], "F1":knn_best["f1"]}
        ])
        st.dataframe(comp)
        fig, ax = plt.subplots()
        cmelt = comp.melt(id_vars=["model"], var_name="metric", value_name="score")
        sns.barplot(data=cmelt, x="metric", y="score", hue="model", ax=ax)
        ax.set_ylim(0,1); ax.set_title("TF-IDF vs KNN best-k")
        st.pyplot(fig)
    except Exception:
        st.info("Không đủ metrics để vẽ so sánh.")

def page_gmail():
    st.header("📧 Quét Gmail & Corrections")
    st.markdown("Luồng: **OAuth → Quét email thật → Phân loại → Gán nhãn SPAM/INBOX → Sửa nhãn → Lưu `corrections.json`**")

    colA,colB = st.columns([1,1])
    model_choice = colA.selectbox("Chọn mô hình", ["TF-IDF + SVM", "KNN + E5 Embedding"])
    max_emails = colB.slider("Số email cần quét", 5, 50, 10)

    if st.button("🔐 Kết nối Gmail & Quét"):
        try:
            service = get_gmail_service()
            try:
                items = get_email_list_with_ids(service, max_results=max_emails)
            except Exception:
                # back-compat
                snippets = get_email_list(service, max_results=max_emails)
                items = [{"id": None, "snippet": s} for s in snippets]

            st.success(f"Đã lấy {len(items)} email")
            for i, it in enumerate(items, 1):
                msg_id = it.get("id")
                text = it.get("snippet","")
                with st.expander(f"Email {i} — id: {msg_id}"):
                    st.write(text)
                    try:
                        pred = predict_tfidf(text) if model_choice.startswith("TF") else predict_embedding(text)
                    except Exception as e:
                        pred = f"error: {e}"
                    st.write(f"**Kết quả:** `{pred}`")

                    c1,c2,c3 = st.columns(3)
                    if c1.button("🏷️ Gán label AI_CORRECTED", key=f"lab_{msg_id}"):
                        try:
                            lid = ensure_label(service, "AI_CORRECTED"); add_label(service, msg_id, lid)
                            st.success("Đã gán label.")
                        except Exception as e:
                            st.warning(f"Không gán được label: {e}")
                    if c2.button("🗂️ Chuyển INBOX", key=f"inb_{msg_id}"):
                        try: move_message(service, msg_id, to_spam=False); st.success("Đã chuyển INBOX.")
                        except Exception as e: st.warning(f"Lỗi move INBOX: {e}")
                    if c3.button("🧹 Chuyển SPAM", key=f"spm_{msg_id}"):
                        try: move_message(service, msg_id, to_spam=True); st.success("Đã chuyển SPAM.")
                        except Exception as e: st.warning(f"Lỗi move SPAM: {e}")

                    st.markdown("**Sửa nhãn (Correction):**")
                    new_label = st.radio("Chọn nhãn đúng", ["spam","ham"], key=f"corr_{msg_id}", horizontal=True)
                    if st.button("💾 Lưu Correction", key=f"save_{msg_id}"):
                        save_correction(text, new_label); st.success("Đã lưu correction.")

        except Exception as e:
            st.error(f"Gmail error: {e}")
            st.info("Hãy đặt `credentials.json` tại thư mục gốc và bật Gmail API.")

    st.markdown("---")
    st.subheader("📋 Corrections đã lưu")
    st.dataframe(get_corrections_df())

# ---------- ROUTER ----------
if page == "🏠 Trang chủ":
    page_home()
elif page == "📊 Tổng quan & Thống kê":
    page_overview()
elif page == "🔍 Phân tích dữ liệu & t-SNE":
    page_analysis_tsne()
elif page == "🧪 Đánh giá mô hình":
    page_eval()
elif page == "📧 Quét Gmail & Corrections":
    page_gmail()
