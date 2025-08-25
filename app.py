import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# Import classifiers
from email_handler import tfidf_classifier, embedding_classifier
from email_handler.data_loader import load_data
from email_handler.corrections_handler import (
    count_corrections,
    save_correction,
    get_corrections_df,
)

st.set_page_config(page_title="Smart Email Classifier", layout="wide")

# ==================== SIDEBAR MENU ====================
st.sidebar.title("📌 MENU")
page = st.sidebar.radio("Chọn chức năng", [
    "📊 SỐ LƯỢNG",
    "🔎 PHÂN TÍCH DỮ LIỆU",
    "🧪 ĐÁNH GIÁ BỘ PHÂN LOẠI",
    "📧 QUÉT EMAIL",
    "✏️ QUẢN LÝ CORRECTION"
])

# Load dataset
df = load_data("data/emails.csv")


# ==================== PAGE 1: SỐ LƯỢNG ====================
if page == "📊 SỐ LƯỢNG":
    st.header("📊 Tổng quan Email Dataset")

    total = len(df)
    spam_count = (df["label"] == "spam").sum()
    ham_count = (df["label"] == "ham").sum()
    corrections = count_corrections()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số Email", total)
    col2.metric("Spam", spam_count)
    col3.metric("Ham", ham_count)
    col4.metric("Corrections", corrections)


# ==================== PAGE 2: PHÂN TÍCH DỮ LIỆU ====================
elif page == "🔎 PHÂN TÍCH DỮ LIỆU":
    st.header("🔎 Phân tích dữ liệu")

    st.subheader("1️⃣ Tổng quan")
    st.write(df.head())

    st.subheader("2️⃣ Phân phối Spam và Ham")
    fig, ax = plt.subplots()
    sns.countplot(x="label", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("3️⃣ Minh hoạ Embedding với t-SNE (1000 mẫu)")
    sample = df.sample(min(1000, len(df)), random_state=42)
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(max_features=200).fit_transform(sample["text"])
    X_emb = TSNE(n_components=2, random_state=42).fit_transform(X.toarray())

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=sample["label"], ax=ax, palette="Set1")
    st.pyplot(fig)


# ==================== PAGE 3: ĐÁNH GIÁ BỘ PHÂN LOẠI ====================
elif page == "🧪 ĐÁNH GIÁ BỘ PHÂN LOẠI":
    st.header("🧪 Đánh giá bộ phân loại")

    st.subheader("1️⃣ Naive Bayes")
    report_nb, cm_nb = tfidf_classifier.evaluate_naive_bayes()
    st.text(report_nb)
    st.write("Confusion Matrix:")
    st.write(cm_nb)

    st.subheader("2️⃣ SVM")
    report_svm, cm_svm = tfidf_classifier.evaluate_svm()
    st.text(report_svm)
    st.write("Confusion Matrix:")
    st.write(cm_svm)


# ==================== PAGE 4: QUÉT EMAIL ====================
elif page == "📧 QUÉT EMAIL":
    st.header("📧 Quét Email từ Gmail")

    st.info("⚠️ Chức năng demo – cần tích hợp Gmail API với OAuth 2.0")

    st.subheader("Cài đặt quét email")
    model_choice = st.selectbox("Chọn bộ phân loại", ["Naive Bayes", "SVM"])
    max_emails = st.slider("Số email tối đa", 10, 200, 50)
    query = st.text_input("Custom query (VD: from:abc@gmail.com)")

    if st.button("Quét Email"):
        st.success("✅ Đã quét và phân loại email (demo)")
        st.write("INBOX: 30 | HAM: 25 | SPAM: 5")


# ==================== PAGE 5: QUẢN LÝ CORRECTION ====================
elif page == "✏️ QUẢN LÝ CORRECTION":
    st.header("✏️ Quản lý Corrections")

    st.write("Người dùng có thể sửa nhãn Spam/Ham để model học lại.")

    idx = st.number_input("Chọn email ID để chỉnh sửa", min_value=0, max_value=len(df)-1, step=1)
    st.write("Email:", df.iloc[idx]["text"])
    new_label = st.radio("Chọn nhãn đúng", ["spam", "ham"])

    if st.button("Lưu Correction"):
        save_correction(df.iloc[idx]["text"], new_label)
        st.success(f"✅ Correction lưu thành công: {new_label}")

    st.subheader("📋 Danh sách Corrections đã lưu")
    st.dataframe(get_corrections_df())
