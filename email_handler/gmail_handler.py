import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Gmail API Scopes (đọc email)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Xác thực Gmail và trả về credentials"""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Đảm bảo bạn đã tải credentials.json từ Google Cloud
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Lưu token để đăng nhập lần sau
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

def get_gmail_service():
    """Tạo service để gọi Gmail API"""
    creds = authenticate_gmail()
    return build('gmail', 'v1', credentials=creds)

def get_email_list(service, max_results=10):
    """Lấy danh sách email (snippet) từ Gmail"""
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    
    email_data = []
    for msg in messages:
        msg_detail = service.users().messages().get(userId='me', id=msg['id']).execute()
        snippet = msg_detail.get('snippet', '')
        email_data.append(snippet)
    
    return email_data
