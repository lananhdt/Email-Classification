import os
import pickle
import json
import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Phạm vi Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_credentials():
    creds = None
    token_path = "token.pickle"

    # Nếu đã có token -> load
    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    # Nếu chưa có hoặc hết hạn
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Local: dùng client_secret.json
            if os.path.exists("client_secret.json"):
                flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            else:
                # Streamlit Cloud: đọc từ st.secrets
                creds_info = json.loads(st.secrets["gcp"]["client_secret"])
                flow = InstalledAppFlow.from_client_config(creds_info, SCOPES)

            # Với Cloud không dùng local server, dùng console
            creds = flow.run_local_server(port=0)

        # Lưu token để lần sau khỏi đăng nhập lại
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    return creds

def get_gmail_service():
    creds = get_credentials()
    service = build('gmail', 'v1', credentials=creds)
    return service

def get_email_list(service, max_results=10, query=''):
    results = service.users().messages().list(userId='me', maxResults=max_results, q=query).execute()
    messages = results.get('messages', [])
    emails = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        snippet = msg_data.get('snippet', '')
        emails.append({
            'id': msg['id'],
            'snippet': snippet
        })
    return emails
