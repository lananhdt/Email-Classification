import os
import pickle
import json
import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def _build_flow():
    # Local: ưu tiên client_secret.json
    if os.path.exists("client_secret.json"):
        return InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)

    # Cloud: đọc từ st.secrets
    try:
        creds_info = json.loads(st.secrets["gcp"]["client_secret"])
        return InstalledAppFlow.from_client_config(creds_info, SCOPES)
    except Exception as e:
        raise RuntimeError("Không tìm thấy client credentials. "
                           "Đặt client_secret.json ở root (local) hoặc cấu hình st.secrets['gcp']['client_secret'] (Cloud).") from e

def get_credentials():
    creds = None
    token_path = "token.pickle"

    # Có token -> load
    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    # Nếu chưa có hoặc hết hạn -> tạo mới/refresh
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = _build_flow()
            # Local OK; Cloud đôi khi không mở được local server -> có thể fallback console
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                creds = flow.run_console()

        # Lưu token cho lần sau
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    return creds

def get_gmail_service():
    creds = get_credentials()
    return build('gmail', 'v1', credentials=creds)

def get_email_list_with_ids(service, max_results=10, query=''):
    res = service.users().messages().list(userId='me', maxResults=max_results, q=query).execute()
    msgs = res.get('messages', [])
    out = []
    for m in msgs:
        data = service.users().messages().get(userId='me', id=m['id']).execute()
        out.append({"id": m["id"], "snippet": data.get("snippet","")})
    return out

def get_email_list(service, max_results=10, query=''):
    res = service.users().messages().list(userId='me', maxResults=max_results, q=query).execute()
    msgs = res.get('messages', [])
    snippets = []
    for m in msgs:
        data = service.users().messages().get(userId='me', id=m['id']).execute()
        snippets.append(data.get('snippet',''))
    return snippets

def ensure_label(service, name: str):
    labels = service.users().labels().list(userId='me').execute().get('labels', [])
    for lb in labels:
        if lb['name'] == name:
            return lb['id']
    created = service.users().labels().create(userId='me', body={
        "name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"
    }).execute()
    return created['id']

def add_label(service, message_id: str, label_id: str):
    service.users().messages().modify(
        userId='me', id=message_id,
        body={"addLabelIds":[label_id], "removeLabelIds":[]}
    ).execute()

def move_message(service, message_id: str, to_spam: bool):
    if to_spam:
        # remove from INBOX, add to SPAM
        service.users().messages().modify(
            userId='me', id=message_id,
            body={"addLabelIds":["SPAM"], "removeLabelIds":["INBOX"]}
        ).execute()
    else:
        # remove SPAM, add INBOX
        service.users().messages().modify(
            userId='me', id=message_id,
            body={"addLabelIds":["INBOX"], "removeLabelIds":["SPAM"]}
        ).execute()
