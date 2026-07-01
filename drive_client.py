"""
Google Drive access: OAuth login + recursive folder walk + image download.

First run opens a browser for you to log into the Google account that has
access to the shared family folders. After that, token.json is cached so
it won't ask again until the token expires.
"""
import io
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDS_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = os.environ.get("GOOGLE_TOKEN_PATH", "token.json")

IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/heic", "image/webp")
FOLDER_MIME = "application/vnd.google-apps.folder"

FILE_FIELDS = "id, name, mimeType, webViewLink, thumbnailLink"


def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def _list_children(service, folder_id):
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields=f"nextPageToken, files({FILE_FIELDS})",
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )
        for f in resp.get("files", []):
            yield f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def walk_images(service, folder_id, path=""):
    """
    Recursively yields dicts for every image file found under folder_id,
    with a human-readable 'folder_path' added for context in search results.
    """
    for item in _list_children(service, folder_id):
        item_path = f"{path}/{item['name']}" if path else item["name"]
        if item["mimeType"] == FOLDER_MIME:
            yield from walk_images(service, item["id"], item_path)
        elif item["mimeType"].startswith(IMAGE_MIME_PREFIXES):
            item["folder_path"] = path
            yield item


def download_file_bytes(service, file_id) -> bytes:
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()