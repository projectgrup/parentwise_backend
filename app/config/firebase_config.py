import firebase_admin
from firebase_admin import credentials
import os
import json

def initialize_firebase():
    if not firebase_admin._apps:
        firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
        if not firebase_credentials_json:
            raise ValueError("FIREBASE_CREDENTIALS environment variable not found")

        cred = credentials.Certificate(json.loads(firebase_credentials_json))
        firebase_admin.initialize_app(cred)
