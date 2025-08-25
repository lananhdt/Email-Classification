import pandas as pd
import os

def load_data(path="data/emails.csv"):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["text","label"])
    df = pd.read_csv(path)
    # Chuẩn hoá cột
    if "text" not in df.columns:
        # đoán tên cột
        for c in df.columns:
            if "text" in c.lower() or "content" in c.lower() or "body" in c.lower():
                df = df.rename(columns={c:"text"})
                break
    if "label" not in df.columns:
        for c in df.columns:
            if "label" in c.lower() or "class" in c.lower() or "target" in c.lower():
                df = df.rename(columns={c:"label"})
                break
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().replace({"spam":"spam","ham":"ham","1":"spam","0":"ham"})
        df = df[df["label"].isin(["spam","ham"])]
    return df.dropna(subset=["text"]).reset_index(drop=True)
