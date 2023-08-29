from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from utils.Api_Authentication import get_api_token
from utils.License_Plate import extract_license_plate_number
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#################################### Start License plate number extraction endpoint ###########################################
@app.post("/license-plate-number-extraction/")
async def read_text_from_image(image_data: dict = Body(...), api_token:str=Depends(get_api_token)):
    status=False
    plate_number=None
    detail="successfully"
    image_url = image_data.get("image_url")
    if not image_url:
        # detail="Image URL is missing in the request body"
        # return {"status":status,"plateNumber":plate_number,"detail":detail}
        raise HTTPException(status_code=400, detail="Image URL is missing in the request body")
    try:
        status,plate_number,detail = extract_license_plate_number(image_url)
        return {"status":status,"plateNumber":plate_number,"detail":detail}
    except HTTPException as exc:
        detail=exc.detail
        return {"status":status,"plateNumber":plate_number,"detail":detail}
    
####################################End License plate number extraction endpoint ###########################################