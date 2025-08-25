import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']  # modify để gán label/di chuyển

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_gmail_service():
    creds = authenticate_gmail()
    return build('gmail', 'v1', credentials=creds)

# ---- Reads ----
def get_email_list(service, max_results=10):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    out = []
    for m in messages:
        msg = service.users().messages().get(userId='me', id=m["id"]).execute()
        out.append(msg.get("snippet",""))
    return out

def get_email_list_with_ids(service, max_results=10):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    out = []
    for m in messages:
        msg = service.users().messages().get(userId='me', id=m["id"]).execute()
        out.append({"id": m["id"], "snippet": msg.get("snippet","")})
    return out

# ---- Labels & Move ----
def ensure_label(service, label_name="AI_CORRECTED"):
    labels = service.users().labels().list(userId='me').execute().get("labels", [])
    for lb in labels:
        if lb["name"] == label_name:
            return lb["id"]
    created = service.users().labels().create(userId='me', body={"name": label_name}).execute()
    return created["id"]

def add_label(service, msg_id, label_id):
    body = {"addLabelIds":[label_id], "removeLabelIds":[]}
    service.users().messages().modify(userId='me', id=msg_id, body=body).execute()

def move_message(service, msg_id, to_spam=True):
    if to_spam:
        body = {"addLabelIds":["SPAM"], "removeLabelIds":["INBOX"]}
    else:
        body = {"addLabelIds":["INBOX"], "removeLabelIds":["SPAM"]}
    service.users().messages().modify(userId='me', id=msg_id, body=body).execute()
