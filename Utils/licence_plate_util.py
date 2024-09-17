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

def extract_license_plate_number(image_url):
    status=False
    plate_number=None
    detail="Number plate pattern not found."
    try:
        response = requests.get(image_url)
    except:
        raise HTTPException(status_code=400, detail=f"Invalid {image_url} url. Status code: {400}")
    
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image from {image_url}. Status code: {response.status_code}")
    
    image_data = BytesIO(response.content)

    try:

        # number_plate_pattern = re.compile(r'\d+-[A-Za-z0-9]+-\d+')
        number_plate_pattern = re.compile(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{2,}(?:[ -][A-Za-z\d]{1,}){0,2}$')
        with Image.open(image_data) as img:
            img_np = np.array(img)
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            result = ocr_model.ocr(img_cv2, cls=True) ### model prediction

            for line in result:
                if not status:
                    for item in line:
                        text = item[1][0]
                        score = item[1][1]
                        if score>0.70:
                            match = number_plate_pattern.search(text)
                            if match:
                                status=True
                                plate_number=text
                                detail="successfully"
                                break
                else:
                    break
        return status,plate_number,detail
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"image not process-able {image_url}. Status code: {response.status_code}")
    finally:
        image_data.close()