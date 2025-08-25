import os
import numpy as np
import pandas as pd
import streamlit as st

# Charts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px

# ====== Domain imports ======
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
    get_gmail_service, get_email_list_with_ids, get_email_list,
    ensure_label, add_label, move_message
)

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Email Classifier", page_icon="📧", layout="wide")

# ================== THEME/CSS ==================
st.markdown("""
<style>
:root {
  --card-bg: rgba(255,255,255,0.04);
  --card-br: 18px;
  --card-bd: 1px solid rgba(255,255,255,0.08);
}
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
.big-title{
  font-size: 44px; font-weight: 900; line-height: 1.1;
  background: linear-gradient(90deg,#a7f3d0,#22c55e);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: .25rem;
}
.subtitle{ color:#cbd5e1; font-size:16px; margin-bottom: 28px; }
.metric{ background: var(--card-bg); border: var(--card-bd);
  border-radius: 16px; padding: 14px; }
.metric .label{ color:#9ca3af; font-size:12px; }
.metric .value{ font-size:26px; font-weight:800; }
.stButton>button{ border-radius: 14px; padding: .75rem 1rem; font-weight:700; }
hr{ border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 16px 0 8px 0;}
</style>
""", unsafe_allow_html=True)

# ================== SESSION NAV ==================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Trang chủ"

PAGES = [
    "🏠 Trang chủ",
    "📊 Phân tích dữ liệu & Thống kê",
    "🧪 Đánh giá mô hình",
    "📧 Quét Gmail & Corrections",
]

st.sidebar.title("📌 MENU")
choice = st.sidebar.radio("Chọn chức năng", PAGES, index=PAGES.index(st.session_state.page))
st.session_state.page = choice

def goto(page_name:str):
    st.session_state.page = page_name
    st.rerun()

# ================== DATA ==================
DATA_PATH = "data/emails.csv"
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.warning(f"Không thể load dữ liệu từ {DATA_PATH}: {e}")
    df = pd.DataFrame(columns=["text","label"])

# ================== HELPERS ==================
def get_stats(_df: pd.DataFrame):
    total = len(_df)
    spam = int((_df["label"]=="spam").sum()) if "label" in _df else 0
    ham  = int((_df["label"]=="ham").sum()) if "label" in _df else 0
    try:
        corr = count_corrections()
    except Exception:
        corr = 0
    return total, spam, ham, corr

@st.cache_data(show_spinner=False)
def tsne_embed(texts: pd.Series, labels: pd.Series, max_samples=1000):
    sample = pd.DataFrame({"text":texts, "label":labels}).dropna()
    sample = sample.sample(min(max_samples, len(sample)), random_state=42)
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(max_features=400).fit_transform(sample["text"])
    X_emb = TSNE(n_components=2, random_state=42, init="random", perplexity=30).fit_transform(X.toarray())
    out = pd.DataFrame({"x": X_emb[:,0], "y": X_emb[:,1], "label": sample["label"].values})
    return out

def get_items_safe(service, max_emails=10, query=""):
    try:
        items = get_email_list_with_ids(service, max_results=max_emails, query=query)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return items
    except Exception:
        pass
    try:
        snippets = get_email_list(service, max_results=max_emails, query=query)
        return [{"id": None, "snippet": s} for s in snippets]
    except Exception:
        return []

def cm_plot(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)

# ================== PAGES ==================
def page_home():
    st.markdown('<div class="big-title">Email Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Khám phá và phân loại email với giao diện tương tác!</div>', unsafe_allow_html=True)

    total, spam, ham, corr = get_stats(df)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric"><div class="label">Tổng số Email</div><div class="value">{total:,}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric"><div class="label">Email Spam</div><div class="value">{spam:,}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric"><div class="label">Email Ham</div><div class="value">{ham:,}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric"><div class="label">Corrections</div><div class="value">{corr:,}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Tính năng")
    a,b,c = st.columns(3)
    if a.button("📊 Phân tích & Thống kê", use_container_width=True):
        goto("📊 Phân tích dữ liệu & Thống kê")
    if b.button("🧪 Đánh giá Mô hình", use_container_width=True):
        goto("🧪 Đánh giá mô hình")
    if c.button("📧 Quét Gmail", use_container_width=True):
        goto("📧 Quét Gmail & Corrections")

def page_analysis_overview():
    st.header("📊 Phân tích Dữ liệu & Thống kê")

    if df.empty:
        st.info("Dataset trống.")
        return

    # — Phân phối Spam/Ham (Plotly)
    st.subheader("Phân phối Spam vs Ham")
    dist = df["label"].value_counts().rename_axis("label").reset_index(name="count")
    fig = px.bar(dist, x="label", y="count", text="count",
                 color="label", color_discrete_map={"ham":"#22c55e","spam":"#ef4444"},
                 labels={"label":"Loại","count":"Số lượng Email"},
                 title="Phân phối Email Spam và Ham")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # — t-SNE
    st.subheader("Minh hoạ embedding với t-SNE (1.000 mẫu)")
    with st.spinner("Đang tính toán t-SNE..."):
        emb = tsne_embed(df["text"], df["label"], max_samples=1000)
    fig2 = px.scatter(emb, x="x", y="y", color="label",
                      color_discrete_map={"ham":"#22c55e","spam":"#ef4444"},
                      labels={"x":"Dim 1","y":"Dim 2","label":"Nhóm"},
                      title="Phân tán embedding qua t-SNE")
    st.plotly_chart(fig2, use_container_width=True)

def page_evaluate_models():
    st.header("🧪 Đánh giá Mô hình")

    # SVM
    st.subheader("TF-IDF + SVM (LinearSVC)")
    svm_report, svm_cm, svm_metrics = None, None, None
    try:
        try:
            svm_report, svm_cm, svm_metrics = evaluate_svm(return_metrics=True)
        except TypeError:
            svm_report, svm_cm = evaluate_svm(); svm_metrics = {}
        st.text(svm_report); cm_plot(svm_cm, "Confusion Matrix — TF-IDF + SVM")
    except Exception as e:
        st.error(f"Lỗi SVM: {e}")

    # Naive Bayes
    st.subheader("Naive Bayes (MultinomialNB)")
    try:
        try:
            nb_report, nb_cm, _ = evaluate_naive_bayes(return_metrics=True)
        except TypeError:
            nb_report, nb_cm = evaluate_naive_bayes()
        st.text(nb_report); cm_plot(nb_cm, "Confusion Matrix — Naive Bayes")
    except Exception as e:
        st.error(f"Lỗi NB: {e}")

    # KNN + FAISS + E5 — k∈{1,3,5}
    st.subheader("KNN + FAISS + E5 (k ∈ {1,3,5})")
    try:
        metrics_df, best_k, cms = evaluate_knn(k_list=[1,3,5])
        st.dataframe(metrics_df, use_container_width=True)

        # Lineplot theo k
        m = metrics_df.melt(id_vars=["k"], var_name="metric", value_name="score")
        fig = px.line(m, x="k", y="score", color="metric", markers=True,
                      title="So sánh chỉ số KNN theo k", range_y=[0,1])
        st.plotly_chart(fig, use_container_width=True)

        # CM theo k
        st.markdown("**Confusion Matrix KNN theo k**")
        ccols = st.columns(len(cms))
        for (k, cm), area in zip(sorted(cms.items(), key=lambda x:x[0]), ccols):
            with area:
                cm_plot(cm, f"k={k}")
    except Exception as e:
        st.error(f"Lỗi KNN: {e}")

    # So sánh TF-IDF vs KNN best-k
    st.subheader("So sánh TF-IDF vs KNN (best-k)")
    try:
        if svm_metrics:
            knn_best = metrics_df[metrics_df["k"]==best_k].iloc[0]
            comp = pd.DataFrame([
                {"model":"TF-IDF+SVM", "Accuracy":svm_metrics.get("accuracy",np.nan),
                 "Precision":svm_metrics.get("precision",np.nan),
                 "Recall":svm_metrics.get("recall",np.nan),
                 "F1":svm_metrics.get("f1",np.nan)},
                {"model":f"KNN (k={best_k})", "Accuracy":knn_best["accuracy"],
                 "Precision":knn_best["precision"], "Recall":knn_best["recall"], "F1":knn_best["f1"]},
            ])
            st.dataframe(comp, use_container_width=True)
            cmelt = comp.melt(id_vars=["model"], var_name="metric", value_name="score")
            figb = px.bar(cmelt, x="metric", y="score", color="model", barmode="group",
                          range_y=[0,1], title="TF-IDF vs KNN best-k")
            st.plotly_chart(figb, use_container_width=True)
        else:
            st.info("Chưa đủ metrics từ SVM để so sánh.")
    except Exception as e:
        st.info(f"Không thể vẽ so sánh: {e}")

def page_gmail_and_corrections():
    st.header("📧 Quét Gmail & Corrections")
    st.caption("Luồng: OAuth → Quét email thật → Phân loại → Gán nhãn SPAM/INBOX → Sửa nhãn → Lưu `corrections.json`")

    left, right = st.columns([2,1])
    with left:
        model_choice = st.selectbox("Chọn bộ phân loại",
                                    ["TF-IDF + SVM (Nhanh – baseline)",
                                     "KNN với Embeddings (Độ chính xác cao)"])
        max_emails = st.number_input("Số email tối đa", 1, 100, 10)
        builtin_query = st.selectbox("Loại email", ["", "is:unread", "in:inbox", "in:spam"])
    with right:
        custom_query = st.text_input("Hoặc nhập custom query", placeholder="from:example.com OR subject:urgent")

    query = (" ".join([builtin_query, custom_query])).strip()

    if st.button("🔐 Kết nối Gmail & Quét Emails", use_container_width=True):
        try:
            service = get_gmail_service()
            items = get_items_safe(service, max_emails=max_emails, query=query)
            st.success(f"Đã lấy {len(items)} email.")
            st.write(f"Query đang dùng: `{query or '(mặc định)'}`")

            # Lưới 3 cột: INBOX | MID | SPAM
            c_inb, c_mid, c_spm = st.columns(3)
            inbox_cnt = spam_cnt = 0

            for i, it in enumerate(items, 1):
                msg_id = it.get("id")
                text = it.get("snippet","").strip() or "(empty snippet)"
                pred = "ham"
                try:
                    if model_choice.startswith("TF"):
                        pred = predict_tfidf(text)
                    else:
                        pred = predict_embedding(text)
                except Exception as e:
                    pred = f"error: {e}"

                target_col = c_spm if str(pred).lower()=="spam" else c_inb
                if str(pred).lower()=="spam": spam_cnt += 1
                else: inbox_cnt += 1

                with target_col:
                    with st.container(border=True):
                        st.markdown(f"**{('SPAM' if pred=='spam' else 'INBOX')}** · id: `{msg_id}`")
                        st.write(text)

                        a,b,c = st.columns(3)
                        if a.button("🏷️ AI_CORRECTED", key=f"lab_{msg_id}_{i}"):
                            try:
                                lid = ensure_label(service, "AI_CORRECTED")
                                if msg_id: add_label(service, msg_id, lid)
                                st.success("Đã gán label.")
                            except Exception as e:
                                st.warning(f"Lỗi label: {e}")

                        if b.button("➡️ INBOX", key=f"inb_{msg_id}_{i}"):
                            try:
                                if msg_id: move_message(service, msg_id, to_spam=False)
                                st.success("Đã chuyển INBOX.")
                            except Exception as e:
                                st.warning(f"Lỗi move: {e}")

                        if c.button("🚫 SPAM", key=f"spm_{msg_id}_{i}"):
                            try:
                                if msg_id: move_message(service, msg_id, to_spam=True)
                                st.success("Đã chuyển SPAM.")
                            except Exception as e:
                                st.warning(f"Lỗi move: {e}")

                        # Correction thủ công
                        new_label = st.radio("Sửa nhãn", ["ham","spam"], horizontal=True, key=f"corr_{msg_id}_{i}")
                        if st.button("💾 Lưu Correction", key=f"save_{msg_id}_{i}"):
                            try:
                                save_correction(text, new_label)
                                st.success("Đã lưu correction.")
                            except Exception as e:
                                st.error(f"Lỗi lưu: {e}")

            with c_mid:
                st.metric("INBOX (dự đoán)", inbox_cnt)
                st.metric("SPAM (dự đoán)", spam_cnt)

        except Exception as e:
            st.error(f"Gmail error: {e}")
            st.info("Local: dùng client_secret.json. Cloud: dùng Secrets. Nếu chạy Cloud, tạo token local một lần rồi upload token.pickle.")

    st.markdown("---")
    st.subheader("📋 Corrections đã lưu")
    try:
        st.dataframe(get_corrections_df(), use_container_width=True)
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
