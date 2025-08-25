import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

from sklearn.manifold import TSNE
from email_handler.data_loader import load_data
from email_handler.tfidf_classifier import evaluate_svm, evaluate_naive_bayes
from email_handler.embedding_classifier import evaluate_knn_multiple_k
from email_handler.corrections_handler import (
    count_corrections,
    save_correction,
    get_corrections_df,
)
from email_handler.gmail_handler import get_gmail_service, get_email_list

# ==================== GLOBAL SETTINGS ====================
st.set_page_config(page_title="Smart Email Classifier", layout="wide")
sns.set_theme(style="darkgrid")

# Init session state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Trang chủ"

PAGES = [
    "🏠 Trang chủ",
    "📊 Phân tích dữ liệu & Thống kê",
    "🧪 Đánh giá mô hình",
    "📧 Gmail & Corrections",
]

# Safe rerun handler
def goto(page_name: str):
    st.session_state.goto = page_name

# ==================== LOAD DATA ====================
df = load_data("data/emails.csv")

# ==================== PAGE: HOME ====================
def page_home():
    st.title("📬 Smart Email Classifier")
    st.subheader("Hệ thống phân loại Email Spam/Ham với Streamlit")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tổng Email", len(df))
        st.metric("Spam", (df["label"] == "spam").sum())
        st.metric("Ham", (df["label"] == "ham").sum())
    with col2:
        st.metric("Corrections", count_corrections())

    st.markdown("---")
    st.subheader("🚀 Tính năng")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 Phân tích dữ liệu & Thống kê", use_container_width=True):
            goto("📊 Phân tích dữ liệu & Thống kê")
        if st.button("🧪 Đánh giá mô hình", use_container_width=True):
            goto("🧪 Đánh giá mô hình")
    with c2:
        if st.button("📧 Gmail & Corrections", use_container_width=True):
            goto("📧 Gmail & Corrections")

# ==================== PAGE: ANALYSIS ====================
def page_analysis():
    st.title("📊 Phân tích dữ liệu & Thống kê")
    st.write("**Hiển thị dữ liệu & biểu đồ trực quan**")

    st.subheader("📌 1. Tổng quan dữ liệu")
    st.dataframe(df.head())

    st.subheader("📌 2. Phân phối Spam và Ham")
    fig, ax = plt.subplots()
    sns.countplot(x="label", data=df, ax=ax, palette="Set1")
    st.pyplot(fig)

    st.subheader("📌 3. Minh họa t-SNE (sample)")
    sample = df.sample(min(1000, len(df)), random_state=42)
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(max_features=200).fit_transform(sample["text"])
    X_emb = TSNE(n_components=2, random_state=42).fit_transform(X.toarray())

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1], hue=sample["label"], ax=ax, palette="Set1")
    st.pyplot(fig)

# ==================== PAGE: EVALUATION ====================
def page_evaluation():
    st.title("🧪 Đánh giá mô hình")

    st.subheader("1️⃣ TF-IDF + Naive Bayes")
    report_nb, cm_nb = evaluate_naive_bayes()
    st.text(report_nb)
    st.write("Confusion Matrix:")
    st.write(cm_nb)

    st.subheader("2️⃣ TF-IDF + SVM")
    report_svm, cm_svm = evaluate_svm()
    st.text(report_svm)
    st.write("Confusion Matrix:")
    st.write(cm_svm)

    st.subheader("3️⃣ FAISS + KNN (Embedding)")
    k_values = [1, 3, 5]
    results_knn = evaluate_knn_multiple_k(k_values)

    st.write("Hiệu suất theo k:")
    df_results = pd.DataFrame(results_knn)
    st.dataframe(df_results)

    fig, ax = plt.subplots()
    sns.lineplot(data=df_results, x="k", y="accuracy", marker="o", ax=ax)
    st.pyplot(fig)

# ==================== PAGE: GMAIL & CORRECTIONS ====================
def page_gmail():
    st.title("📧 Gmail & Corrections")

    st.subheader("🔍 Quét Email từ Gmail")
    max_emails = st.slider("Số email tối đa", 10, 100, 20)
    query = st.text_input("Custom query (VD: from:abc@gmail.com)")
    if st.button("Quét Email"):
        try:
            service = get_gmail_service()
            emails = get_email_list(service, max_results=max_emails, query=query)
            st.success(f"✅ Đã lấy {len(emails)} email")
            for e in emails:
                st.write(f"- {e['snippet']}")
        except Exception as ex:
            st.error(f"Lỗi Gmail API: {ex}")

    st.markdown("---")
    st.subheader("✏️ Corrections")
    idx = st.number_input("Chọn email ID", min_value=0, max_value=len(df)-1, step=1)
    st.write("Email:", df.iloc[idx]["text"])
    new_label = st.radio("Chọn nhãn đúng", ["spam", "ham"])
    if st.button("Lưu Correction"):
        save_correction(df.iloc[idx]["text"], new_label)
        st.success(f"✅ Correction lưu thành công: {new_label}")

    st.write("Danh sách Corrections:")
    st.dataframe(get_corrections_df())

# ==================== RENDER ====================
choice = st.sidebar.radio("📌 Chọn trang", PAGES, index=PAGES.index(st.session_state.page))
st.session_state.page = choice

if st.session_state.page == "🏠 Trang chủ":
    page_home()
elif st.session_state.page == "📊 Phân tích dữ liệu & Thống kê":
    page_analysis()
elif st.session_state.page == "🧪 Đánh giá mô hình":
    page_evaluation()
elif st.session_state.page == "📧 Gmail & Corrections":
    page_gmail()

# Safe rerun after button navigation
if "goto" in st.session_state:
    st.session_state.page = st.session_state.goto
    del st.session_state.goto
    st.experimental_rerun()
