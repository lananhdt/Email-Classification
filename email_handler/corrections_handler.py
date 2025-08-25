import os
import json
import pandas as pd

# Nơi lưu corrections
CORRECTIONS_FILE = "data/corrections.json"


def load_corrections():
    """Load corrections từ file JSON"""
    if not os.path.exists(CORRECTIONS_FILE):
        return []
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_correction(email_text, correct_label):
    """Thêm correction mới và lưu vào file"""
    corrections = load_corrections()
    corrections.append({"text": email_text, "label": correct_label})

    os.makedirs("data", exist_ok=True)
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)


def count_corrections():
    """Đếm số lượng corrections hiện có"""
    return len(load_corrections())


def get_corrections_df():
    """Trả về corrections dưới dạng pandas DataFrame"""
    data = load_corrections()
    if not data:
        return pd.DataFrame(columns=["text", "label"])
    return pd.DataFrame(data)
