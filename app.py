import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report, confusion_matrix
from email_handler import data_loader, tfidf_classifier, embedding_classifier

st.set_page_config(page_title="Email Classifier", layout="wide")

# ---------------- Sidebar ----------------
menu = ["📊 Overview", "🔍 Data Analysis", "🤖 Model Evaluation", "📧 Gmail & Correction"]
choice = st.sidebar.radio("Chọn trang", menu)

# ---------------- Page 1: Overview ----------------
if choice == "📊 Overview":
    st.title("📊 Email Classifier - Tổng quan")

    stats = data_loader.load_stats()
    st.write("### Thống kê dữ liệu")
    st.json(stats)

    fig, ax = plt.subplots()
    ax.pie([stats['spam'], stats['ham']], labels=['Spam', 'Ham'], autopct='%1.1f%%')
    ax.set_title("Tỉ lệ Spam / Ham")
    st.pyplot(fig)

# ---------------- Page 2: Data Analysis ----------------
elif choice == "🔍 Data Analysis":
    st.title("🔍 Phân tích dữ liệu")

    data = data_loader.load_data()
    st.write("### Một vài email mẫu")
    st.dataframe(data.head())

    # Word length distribution
    data['length'] = data['text'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(data['length'], bins=50, ax=ax)
    ax.set_title("Phân bố độ dài email")
    st.pyplot(fig)

# ---------------- Page 3: Model Evaluation ----------------
elif choice == "🤖 Model Evaluation":
    st.title("🤖 Đánh giá mô hình")

    model_type = st.selectbox("Chọn mô hình", ["Naive Bayes", "TF-IDF + SVM", "KNN + Embedding"])
    
    if st.button("Chạy đánh giá"):
        if model_type == "Naive Bayes":
            report, cm = tfidf_classifier.evaluate_naive_bayes()
        elif model_type == "TF-IDF + SVM":
            report, cm = tfidf_classifier.evaluate_svm()
        else:
            report, cm = embedding_classifier.evaluate_knn()

        st.text("### Classification Report")
        st.text(report)

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# ---------------- Page 4: Gmail & Correction ----------------
elif choice == "📧 Gmail & Correction":
    st.title("📧 Gmail & Correction")
    st.write("Kết nối Gmail API, phân loại email thật, cho phép người dùng sửa nhãn.")

    emails = data_loader.load_gmail_emails(max_results=10)
    for idx, row in emails.iterrows():
        with st.expander(f"Email {idx+1}: {row['subject']}"):
            st.write(row['body'][:300] + "...")
            pred = tfidf_classifier.predict_single(row['subject'] + " " + row['body'])
            st.write(f"**Dự đoán:** {pred}")

            correction = st.radio("Phân loại đúng không?", ["Đúng", "Sai"], key=idx)
            if correction == "Sai":
                true_label = st.radio("Chọn nhãn đúng:", ["Spam", "Ham"], key=f"label_{idx}")
                if st.button(f"Lưu correction {idx}"):
                    with open("data/corrections.json", "a") as f:
                        json.dump({"text": row['subject'] + " " + row['body'], "label": true_label}, f)
                        f.write("\n")
                    st.success("Correction đã lưu!")
