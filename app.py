# app.py
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ====== Model & utils (giữ import như bạn dùng trước đó) ======
from email_handler import tfidf_classifier, embedding_classifier
from email_handler.data_loader import load_data
from email_handler.corrections_handler import (
    count_corrections,
    save_correction,
    get_corrections_df,
)

# Dùng trực tiếp các hàm dự đoán nếu bạn đã để ở root
try:
    from tfidf_classifier import predict_tfidf  # baseline TF-IDF
except Exception:
    predict_tfidf = None

try:
    from embedding_classifier import predict_embedding  # E5+FAISS
except Exception:
    predict_embedding = None

# Gmail
from gmail_handler import get_gmail_service, get_email_list

# ================== PAGE CONFIG & THEME ==================
st.set_page_config(page_title="Email Classifier", layout="wide", page_icon="📧")

# CSS nhẹ cho dark glassy UI
st.markdown("""
<style>
/* Tiêu đề gradient */
.big-title {
  font-size: 44px; font-weight: 800;
  background: linear-gradient(90deg,#a7f3d0,#22c55e);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0 0 4px 0;
}
.subtitle { color: #cbd5e1; font-size: 16px; margin-bottom: 24px; }
.metric-note { font-size: 12px; color: #86efac; }
.btn-grid .stButton>button {
  width: 100%; padding: 12px 16px; border-radius: 16px; font-weight: 600;
}
.card {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px; padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE ==================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Trang chủ"

def goto(p): st.session_state.page = p

# ================== DATA ==================
# Bạn có thể đổi đường dẫn tùy repo
DATA_PATH = "data/emails.csv"
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.warning(f"Không thể load dữ liệu từ {DATA_PATH}: {e}")
    df = pd.DataFrame(columns=["text","label"])

# ================== NAVIGATION (sidebar) ==================
st.sidebar.title("📌 MENU")
side_choice = st.sidebar.radio(
    "Điều hướng",
    ["🏠 Trang chủ", "📊 Tổng quan & Thống kê", "🔍 Phân tích & t-SNE",
     "🧪 Đánh giá mô hình", "📧 Gmail & Corrections"]
)
st.session_state.page = side_choice

# ================== HOMEPAGE ==================
def home():
    st.markdown('<div class="big-title">Email Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Khám phá và phân loại email với giao diện tương tác!</div>', unsafe_allow_html=True)

    total = len(df)
    spam_count = int((df["label"] == "spam").sum()) if "label" in df else 0
    ham_count  = int((df["label"] == "ham").sum()) if "label" in df else 0
    try:
        corrections = count_corrections()
    except Exception:
        corrections = 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Tổng số Email", f"{total:,}")
    with c2: st.metric("Email Spam", f"{spam_count:,}", "↑") ; st.caption(" ")
    with c3: st.metric("Email Ham", f"{ham_count:,}", "↑"); st.caption(" ")
    with c4: st.metric("Corrections", f"{corrections:,}")

    st.markdown("#### Tính năng")
    cA,cB,cC,cD = st.columns([1,1,1,1])
    with cA:
        if st.button("🔬 Phân tích Dữ liệu", use_container_width=True): goto("🔍 Phân tích & t-SNE")
    with cB:
        if st.button("🧪 Đánh giá Bộ phân loại", use_container_width=True): goto("🧪 Đánh giá mô hình")
    with cC:
        if st.button("📧 Quét Gmail", use_container_width=True): goto("📧 Gmail & Corrections")
    with cD:
        if st.button("📝 Quản lý Corrections", use_container_width=True): goto("📧 Gmail & Corrections")

# ================== PAGE: OVERVIEW ==================
def page_overview():
    st.header("📊 Tổng quan & Thống kê")
    if df.empty:
        st.info("Dataset trống.")
        return

    st.subheader("Bảng dữ liệu (5 dòng đầu)")
    st.dataframe(df.head())

    st.subheader("Phân phối Spam/Ham")
    fig, ax = plt.subplots()
    sns.countplot(x="label", data=df, ax=ax)
    ax.set_xlabel("Label"); ax.set_ylabel("Count")
    st.pyplot(fig)

# ================== PAGE: ANALYSIS & TSNE ==================
def page_analysis_tsne():
    st.header("🔍 Phân tích dữ liệu & t-SNE")

    if df.empty:
        st.info("Dataset trống.")
        return

    st.subheader("Giảm chiều với t-SNE (demo trên TF-IDF 200 features)")
    sample = df.sample(min(1000, len(df)), random_state=42) if len(df) > 0 else df

    if not sample.empty:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(max_features=200).fit_transform(sample["text"])
        X_emb = TSNE(n_components=2, random_state=42, init="random", perplexity=30).fit_transform(X.toarray())

        fig, ax = plt.subplots()
        sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=sample["label"], ax=ax, palette="Set1", s=18)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(loc="best")
        st.pyplot(fig)
    else:
        st.info("Không đủ dữ liệu để vẽ t-SNE.")

# ================== HELPERS: DRAW CM HEATMAP ==================
def draw_cm(cm, labels=("ham","spam"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig)

# ================== PAGE: EVALUATION ==================
def page_evaluate_models():
    st.header("🧪 Đánh giá mô hình")

    # ---- TF-IDF + SVM ----
    st.subheader("TF-IDF + SVM (LinearSVC) — baseline")
    try:
        report_svm, cm_svm = tfidf_classifier.evaluate_svm()
        st.text(report_svm)
        draw_cm(cm_svm, title="TF-IDF + SVM")
    except Exception as e:
        st.error(f"Lỗi evaluate_svm(): {e}")

    # ---- Naive Bayes ----
    st.subheader("Naive Bayes (MultinomialNB)")
    try:
        report_nb, cm_nb = tfidf_classifier.evaluate_naive_bayes()
        st.text(report_nb)
        draw_cm(cm_nb, title="Naive Bayes")
    except Exception as e:
        st.error(f"Lỗi evaluate_naive_bayes(): {e}")

    # ---- KNN + FAISS + E5 ----
    st.subheader("KNN + FAISS + E5 Embedding (k ∈ {1,3,5})")
    k_list = [1,3,5]
    try:
        # Kỳ vọng module có sẵn evaluate_knn(k_list) → (df_metrics, best_k, cms_by_k)
        eval_out = embedding_classifier.evaluate_knn(k_list=k_list)
        if isinstance(eval_out, tuple) and len(eval_out) >= 2:
            metrics_df = eval_out[0]
            best_k = eval_out[1]
            cms_by_k = eval_out[2] if len(eval_out) > 2 else {}

            st.write("Bảng metrics theo k:")
            st.dataframe(metrics_df)

            # Lineplot cho mỗi metric theo k
            melt_df = metrics_df.melt(id_vars=["k"], var_name="metric", value_name="score")
            fig, ax = plt.subplots()
            sns.lineplot(data=melt_df, x="k", y="score", hue="metric", marker="o", ax=ax)
            ax.set_title("KNN metrics theo k"); ax.set_ylabel("Score"); ax.set_xlabel("k")
            st.pyplot(fig)

            # So sánh TF-IDF vs KNN-best-k bằng Accuracy/Precision/Recall/F1
            try:
                # Lấy từ report SVM (sklearn classification_report -> text); bạn có thể
                # thay bằng tfidf_classifier.get_metrics_svm() nếu đã có.
                # Ở đây demo: giả sử metrics_df có hàng TFIDF
                tfidf_row = None
                if "model" in metrics_df.columns:
                    tfidf_row = metrics_df[metrics_df["model"]=="TFIDF_SVM"].iloc[0].to_dict()
                # fallback: lấy dòng k==best_k cho KNN
                knn_best = metrics_df[metrics_df["k"]==best_k].iloc[0]

                comp = pd.DataFrame([
                    {"model":"TFIDF+SVM","Accuracy":tfidf_row.get("accuracy") if tfidf_row else np.nan,
                     "Precision":tfidf_row.get("precision") if tfidf_row else np.nan,
                     "Recall":tfidf_row.get("recall") if tfidf_row else np.nan,
                     "F1":tfidf_row.get("f1") if tfidf_row else np.nan},
                    {"model":f"KNN (k={best_k})","Accuracy":knn_best.get("accuracy"),
                     "Precision":knn_best.get("precision"),
                     "Recall":knn_best.get("recall"),
                     "F1":knn_best.get("f1")}
                ])
                st.write("So sánh TF-IDF vs KNN best-k:")
                st.dataframe(comp)

                fig, ax = plt.subplots()
                comp_melt = comp.melt(id_vars=["model"], var_name="metric", value_name="score")
                sns.barplot(data=comp_melt, x="metric", y="score", hue="model", ax=ax)
                ax.set_title("TF-IDF vs KNN best-k"); ax.set_ylim(0,1)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Không thể dựng biểu đồ so sánh TF-IDF vs KNN: {e}")

            # Vẽ CM cho từng k nếu có
            for k in k_list:
                if isinstance(cms_by_k, dict) and k in cms_by_k:
                    draw_cm(cms_by_k[k], title=f"KNN (k={k}) — Confusion Matrix")

        else:
            st.warning("embedding_classifier.evaluate_knn(k_list) chưa trả về đúng định dạng.")
    except Exception as e:
        st.error(f"Lỗi khi evaluate KNN: {e}")
        st.caption("Bạn có thể bổ sung hàm `evaluate_knn(k_list)` trong embedding_classifier để trả về (metrics_df, best_k, cms_by_k).")

# ================== PAGE: GMAIL & CORRECTIONS ==================
def page_gmail_and_corrections():
    st.header("📧 Gmail & Corrections")

    st.markdown("**Luồng:** OAuth → Quét email → Phân loại → Gán nhãn SPAM/INBOX → Sửa nhãn → Lưu `corrections.json`")

    colA,colB = st.columns([1,1])
    model_choice = colA.selectbox("Chọn mô hình phân loại", ["TF-IDF + SVM", "KNN + E5 Embedding"])
    max_emails = colB.slider("Số email tối đa cần quét", 5, 50, 10)

    if st.button("🔐 Kết nối Gmail & Quét"):
        try:
            service = get_gmail_service()
            emails = get_gmail_list_with_ids_safe(service, max_emails)

            if not emails:
                st.info("Không lấy được email nào.")
                return

            st.success(f"Đã lấy {len(emails)} email.")
            for i, item in enumerate(emails, 1):
                msg_id = item.get("id")
                text = item.get("snippet","")

                with st.expander(f"Email {i} — id: {msg_id}"):
                    st.write(text)

                    # Phân loại
                    pred = "unknown"
                    try:
                        if model_choice.startswith("TF-IDF") and predict_tfidf:
                            pred = predict_tfidf(text)
                        elif predict_embedding:
                            pred = predict_embedding(text)
                    except Exception as e:
                        st.error(f"Lỗi phân loại: {e}")

                    st.write(f"**Kết quả:** `{pred}`")

                    # Gán nhãn / chuyển hộp thư (nếu gmail_handler đã có hàm)
                    cc1, cc2, cc3 = st.columns([1,1,1])
                    if cc1.button("🏷️ Gán label AI_CORRECTED", key=f"lab_{msg_id}"):
                        try:
                            from gmail_handler import ensure_label, add_label
                            lid = ensure_label(service, "AI_CORRECTED")
                            add_label(service, msg_id, lid)
                            st.success("Đã gán label AI_CORRECTED.")
                        except Exception as e:
                            st.warning(f"Chưa hỗ trợ gán label: {e}")

                    if cc2.button("🗂️ Chuyển INBOX", key=f"inb_{msg_id}"):
                        try:
                            from gmail_handler import move_message
                            move_message(service, msg_id, to_spam=False)
                            st.success("Đã chuyển về INBOX.")
                        except Exception as e:
                            st.warning(f"Chưa hỗ trợ chuyển INBOX: {e}")

                    if cc3.button("🧹 Chuyển SPAM", key=f"spm_{msg_id}"):
                        try:
                            from gmail_handler import move_message
                            move_message(service, msg_id, to_spam=True)
                            st.success("Đã chuyển sang SPAM.")
                        except Exception as e:
                            st.warning(f"Chưa hỗ trợ chuyển SPAM: {e}")

                    # Correction thủ công
                    st.markdown("**Sửa nhãn (Correction):**")
                    new_label = st.radio(
                        "Chọn nhãn đúng", ["spam","ham"], key=f"corr_{msg_id}", horizontal=True
                    )
                    if st.button("💾 Lưu Correction", key=f"save_{msg_id}"):
                        try:
                            save_correction(text, new_label)
                            st.success("Đã lưu correction.")
                        except Exception as e:
                            st.error(f"Lỗi lưu correction: {e}")

        except Exception as e:
            st.error(f"Lỗi Gmail: {e}")
            st.info("Đảm bảo có `credentials.json` và đã bật Gmail API.")

    st.markdown("---")
    st.subheader("📋 Corrections đã lưu")
    try:
        st.dataframe(get_corrections_df())
    except Exception as e:
        st.info(f"Chưa có corrections: {e}")

# ---- helper: cố lấy cả id lẫn snippet dù gmail_handler cũ chỉ trả snippet
def get_gmail_list_with_ids_safe(service, max_results=10):
    """
    Ưu tiên dùng gmail_handler.get_email_list (nếu đã trả về list[dict]),
    nếu chỉ trả snippet -> tự list để lấy id.
    Trả về: list of dict {id, snippet}
    """
    items = []
    try:
        from gmail_handler import get_email_list_with_ids
        items = get_email_list_with_ids(service, max_results=max_results)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return items
    except Exception:
        pass

    # Fallback: dùng API trực tiếp
    try:
        results = service.users().messages().list(userId='me', maxResults=max_results).execute()
        messages = results.get('messages', [])
        out = []
        for m in messages:
            msg = service.users().messages().get(userId='me', id=m["id"]).execute()
            out.append({"id": m["id"], "snippet": msg.get("snippet","")})
        return out
    except Exception:
        # Fallback nữa: dùng get_email_list cũ -> chỉ snippet
        try:
            snippets = get_email_list(service, max_results=max_results)
            return [{"id": None, "snippet": s} for s in snippets]
        except Exception:
            return []

# ================== ROUTER ==================
if st.session_state.page == "🏠 Trang chủ":
    home()
elif st.session_state.page == "📊 Tổng quan & Thống kê":
    page_overview()
elif st.session_state.page == "🔍 Phân tích & t-SNE":
    page_analysis_tsne()
elif st.session_state.page == "🧪 Đánh giá mô hình":
    page_evaluate_models()
elif st.session_state.page == "📧 Gmail & Corrections":
    page_gmail_and_corrections()
