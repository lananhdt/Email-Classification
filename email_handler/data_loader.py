import pandas as pd
import os

# Load dữ liệu gốc (spam/ham dataset CSV)
def load_data(file_path="data/emails.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file {file_path}")
    return pd.read_csv(file_path)

# Trả về thống kê Spam/Ham
def load_stats(file_path="data/emails.csv"):
    df = load_data(file_path)
    stats = {
        "total": len(df),
        "spam": (df['label'] == "spam").sum(),
        "ham": (df['label'] == "ham").sum()
    }
    return stats

# (Option) Kết nối Gmail API và lấy email thật
def load_gmail_emails(max_results=10):
    # TODO: tích hợp Gmail API
    # Tạm trả về DataFrame mock
    data = {
        "subject": ["Win a prize", "Meeting tomorrow", "Limited time offer"],
        "body": [
            "You have won a lottery! Claim now.",
            "Let's meet at 10 AM tomorrow.",
            "Special deal just for you."
        ]
    }
    return pd.DataFrame(data)
