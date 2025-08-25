import os
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def gmail_authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def fetch_emails(max_results=20, query=""):
    service = gmail_authenticate()
    results = service.users().messages().list(userId="me", maxResults=max_results, q=query).execute()
    messages = results.get("messages", [])
    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId="me", id=msg["id"]).execute()
        snippet = txt.get("snippet", "")
        emails.append(snippet)
    return emails
