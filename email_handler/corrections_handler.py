import os, json
import pandas as pd

CORR_PATH = "corrections.json"

def _ensure_file():
    if not os.path.exists(CORR_PATH):
        with open(CORR_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def save_correction(text, label):
    _ensure_file()
    with open(CORR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.append({"text": text, "label": label})
    with open(CORR_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def count_corrections():
    _ensure_file()
    with open(CORR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)

def get_corrections_df():
    _ensure_file()
    with open(CORR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)
