from dotenv.main import load_dotenv
from fastapi import HTTPException
from paddleocr import PaddleOCR
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import cv2
import re


def is_night_captured_image(image_url,threshold=100):
    try:
        response = requests.get(image_url)
    except:
        raise HTTPException(status_code=400, detail=f"Invalid {image_url} url. Status code: {400}")
    
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image from {image_url}. Status code: {response.status_code}")
    
    image_data = BytesIO(response.content)

    try:
        with Image.open(image_data) as img:
            img_np = np.array(img)
            # img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
            avg_brightness = np.mean(gray_image)
            
            if avg_brightness > threshold:
                return True
            else:
                return False
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"image not process-able {image_url}. Status code: {response.status_code}")
    finally:
        image_data.close()