from custom_utils.Car_Damage import random_name_generator,s3_file_downloader, damage_detection_in_image,damage_detection_in_video
from custom_utils.License_Plate import extract_license_plate_number
from custom_utils.S3_bucket import upload_file_to_s3_bucket
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from custom_utils.Api_Authentication import get_api_token
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


@app.post("/damage-detection/")
async def damage_detection(body: dict = Body(...), api_token:str=Depends(get_api_token)):
    dir_name="s3_files/"
    s3_url=body.get("s3_url")
    extension=body.get("extension")
    file_name=random_name_generator()

    if s3_file_downloader(s3_url,file_name+extension):
        processed_file_path=None
        if extension.lower()==".jpg":
            processed_file_path,message=damage_detection_in_image(dir_name,file_name,extension)
        elif extension.lower()==".mp4":
            processed_file_path,message=damage_detection_in_video(dir_name,file_name,extension)
        
        if processed_file_path!=None:
            uploaded_s3_link=upload_file_to_s3_bucket(processed_file_path,file_name,extension)
            ## add comments
        return {"processed_file_path": processed_file_path,"uploaded_s3_link":uploaded_s3_link,"extension":extension,"message":message}