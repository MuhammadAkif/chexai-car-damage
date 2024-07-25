from fastapi import HTTPException
from google.cloud import vision
from io import BytesIO
from PIL import Image
import requests
import re
import os

def extract_license_plate_number(image_url):
    credentials_json= {
    "type": "service_account",
    "project_id": os.environ['GOOGLE_PROJECT_ID'],
    "private_key_id": os.environ['GOOGLE_PRIVATE_KEY_ID'],
    "private_key": os.environ['GOOGLE_PRIVATE_KEY'],
    "client_email": os.environ['GOOGLE_CLIENT_EMAIL'],
    "client_id": os.environ['GOOGLE_CLIENT_ID'],
    "auth_uri": os.environ['GOOGLE_AUTH_URI'],
    "token_uri": os.environ['GOOGLE_TOKEN_URI'],
    "auth_provider_x509_cert_url": os.environ['GOOGLE_AUTH_PROVIDER_X509_CERT_URL'],
    "client_x509_cert_url": os.environ['GOOGLE_CLIENT_X509_CERT_URL'],
    "universe_domain": os.environ['GOOGLE_UNIVERSE_DOMAIN']
    }
    
    client = vision.ImageAnnotatorClient.from_service_account_info(credentials_json)

    # Download the image from the URL
    status=False
    plate_number=None
    detail="successfully"
    response = requests.get(image_url)
    if response.status_code != 200:
        # detail=f"Failed to download image from {image_url}"
        # return status,plate_number,detail
        raise HTTPException(status_code=400, detail=f"Failed to download image from {image_url}. Status code: {response.status_code}")

    # Read the image data
    image = Image.open(BytesIO(response.content))
    content = response.content

    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)

    if response.error.message:
        # detail=f"Error detecting text: {response.error.message}"
        # return status,plate_number,detail
        raise HTTPException(status_code=500, detail=f"Error detecting text: {response.error.message}")
    
    number_plate_pattern = re.compile(r'\d+-[A-Za-z0-9]+-\d+')
    for text in response.text_annotations:
        print("word: ",text.description)
        match = number_plate_pattern.search(text.description)
        if match:
            print("matched: ",match.group()) 
            return True,match.group(),"successfully"
        else:
            return False,None,"plateNumber not found in the image."