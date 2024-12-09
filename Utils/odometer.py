from dotenv.main import load_dotenv
from fastapi import HTTPException
from paddleocr import PaddleOCR
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import cv2
import re
import os


load_dotenv()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
ocr_model = PaddleOCR(use_angle_cls=os.environ['OCR_USE_ANGLE_CLS'], lang='en', use_gpu = os.environ['OCR_USE_GPU'] )

def extract_mileage(image_url):
    status=False
    mileage=None
    detail="Mileage not found."
    try:
        response = requests.get(image_url)
    except:
        raise HTTPException(status_code=400, detail=f"Invalid {image_url} url. Status code: {400}")
    
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image from {image_url}. Status code: {response.status_code}")
    
    image_data = BytesIO(response.content)

    try:

        mileage_pattern=re.compile(r'\b\d+(\.\d+)?\s*(?:mi|mi\s|mi\n)\b', re.IGNORECASE)
        with Image.open(image_data) as img:
            img_np = np.array(img)
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            result = ocr_model.ocr(img_cv2, cls=True) ### model prediction

            # Combine all text into a single sentence
        full_text = ""
        for line in result:
            for item in line:
                text = item[1][0]
                score = item[1][1]
                if score > 0.70:
                    full_text += f"{text} "  # Concatenate all text with spaces

        # Match the pattern on the concatenated string
        match = mileage_pattern.search(full_text)
        if match:
            status = True
            mileage = match.group()  # Extract the matched text
            detail = "successfully extracted pattern"

        return status, mileage, detail
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"image not process-able {image_url}. Status code: {response.status_code}")
    finally:
        image_data.close()