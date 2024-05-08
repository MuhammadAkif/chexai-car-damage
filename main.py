# from custom_utils.Car_Damage import random_name_generator,s3_file_downloader, damage_detection_in_image,damage_detection_in_video
# from custom_utils.Car_Damage2 import damage_detection_in_image2, damage_detection_in_video2
# from custom_utils.License_Plate import extract_license_plate_number
from custom_utils.S3_bucket import upload_file_to_s3_bucket
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from custom_utils.Api_Authentication import get_api_token
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
from pydantic import BaseModel, Field
import json
import os

load_dotenv()


client = OpenAI(api_key=os.environ['OPEN_AI_KEY'])


class ChatRequest(BaseModel):
    model: str = "gpt-4-turbo"
    images_link: List
    image_side: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ################################### Start License plate number extraction endpoint ###########################################
# @app.post("/license-plate-number-extraction/")
# async def read_text_from_image(image_data: dict = Body(...), api_token:str=Depends(get_api_token)):
#     status=False
#     plate_number=None
#     detail="successfully"
#     image_url = image_data.get("image_url")
#     if not image_url:
#         # detail="Image URL is missing in the request body"
#         # return {"status":status,"plateNumber":plate_number,"detail":detail}
#         raise HTTPException(status_code=400, detail="Image URL is missing in the request body")
#     try:
#         status,plate_number,detail = extract_license_plate_number(image_url)
#         return {"status":status,"plateNumber":plate_number,"detail":detail}
#     except HTTPException as exc:
#         detail=exc.detail
#         return {"status":status,"plateNumber":plate_number,"detail":detail}
    
# ####################################End License plate number extraction endpoint ###########################################


# @app.post("/damage-detection/")
# async def damage_detection(body: dict = Body(...), api_token:str=Depends(get_api_token)):
#     dir_name="s3_files/"
#     s3_url=body.get("s3_url")
#     extension=body.get("extension")
#     file_name=random_name_generator()

#     if s3_file_downloader(s3_url,file_name+extension):
#         processed_file_path=None
#         if extension.lower()==".jpg":
#             processed_file_path,message=damage_detection_in_image(dir_name,file_name,extension)
#         elif extension.lower()==".mp4":
#             processed_file_path,message=damage_detection_in_video(dir_name,file_name,extension)
        
#         if processed_file_path!=None:
#             uploaded_s3_link=upload_file_to_s3_bucket(processed_file_path,file_name,extension)
#         return {"processed_file_path": processed_file_path,"uploaded_s3_link":uploaded_s3_link,"extension":extension,"message":message}
        

# @app.post("/damage-detection2/")
# async def damage_detection(body: dict = Body(...), api_token:str=Depends(get_api_token)):
#     dir_name="s3_files/"
#     s3_url=body.get("s3_url")
#     extension=body.get("extension")
#     file_name=random_name_generator()

#     if s3_file_downloader(s3_url,file_name+extension):
#         processed_file_path=None
#         if extension.lower()==".jpg":
#             processed_file_path,message,damage_rectangle,original_img_info=damage_detection_in_image2(dir_name,file_name,extension)
#             if processed_file_path!=None:
#                 uploaded_s3_link=upload_file_to_s3_bucket(processed_file_path,file_name,extension)
#             return {"image_s3_link":s3_url,"processed_img_s3_link":uploaded_s3_link,"extension":extension,"message":message,"org_img_info":original_img_info,"damages":damage_rectangle}
        
#         elif extension.lower()==".mp4":
#             processed_file_path,message=damage_detection_in_video2(dir_name, file_name ,extension)
    
#             if processed_file_path!=None:
#                 uploaded_s3_link=upload_file_to_s3_bucket(processed_file_path,file_name,extension)
#             return {"processed_file_path":file_name+extension,"uploaded_s3_link":uploaded_s3_link,"extension":extension,"message":message}
        
@app.post("/llm-chat/")
async def llm_chat_response(chat_request: ChatRequest, api_token:str=Depends(get_api_token)):

    images_link = chat_request.images_link
    if len(images_link)>2:
        raise HTTPException(status_code=400, detail="You have provided more than two images for comparison.")
    
    print(images_link[0])
    if len(images_link)==1:
        response = client.chat.completions.create(
        model = chat_request.model,
        messages=[
            {
            "role": "user",
            "content": [
                # {
                # "type": "text",
                # "text": "I have provided "+chat_request.image_side+" image of the vehicle. Please check the small spots, small dents and big dents on the vehicle body and also suggest the estimated repare cost based on detected damages in dollar and also add one line comment, if any damage detected return in the following json format\
                #     {'small_dents':2/3 , 'small_spots':3/10,'big_dents':0/4 or any detected big dents, 'scratches':1 or 6 or any detected scratches,'cost':50/200, 'comment':'small dents and spots detected but there is no severe damage detected.'}, and if no the simply return {'small_dents':0, 'small_spots':0,'big_dents':0,'scratches':0,'cost':0, 'comment':'There is no damage found.'} and no cost. Please make sure the response should be 100 persent in the json format any there is no any text defore and end.",
                # },
                {
                "type": "text",
                "text": "I have provided "+chat_request.image_side+" image of the vehicle. Please deeply inspect the whole vehicle if you find any spots, scratches, dents, or any damage and alos focus on small scratches, dent and damages, please write in the comment with locations only 7-8 words and also suggest the expected cost for fixing vehicle and response return in the following json format\
                    {'cost':50/200, 'comment':'2 scratched and 1 dent on the right door./ 4 spots on front door and scratch on window.'}, and if no the simply return {'cost':0, 'comment':'There is no damage found.'} and no cost. Please make sure the response should be 100 percent in the json format any there is no any text before and after/end.",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": images_link[0],
                },
                }
            ],
            }
        ],
        max_tokens=300,
        )

        data_str=response.choices[0].message.content
        data_str = data_str.replace("'", '"')
        try:
            data_json = json.loads(data_str)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=e)
        return data_json
    else:
        response = client.chat.completions.create(
        model = chat_request.model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "I have provided two "+chat_request.image_side+" image of the same vehicle. One picture is from previously inspected and second image is fresh picture Please deeply inspect the whole two vehicles if you find any NEW spots, scratches, dents, or any damage and also focus on NEW small scratches, dent and damages, please write in the comment with locations only 7-8 words and also suggest the expected cost for fixing vehicle and response return in the following json format\
                    {'cost':50/200, 'comment':'2 new scratches and 1 new dent on the right door detected./ 4 snew pots on front door and new scratch on window detected.'}, and if no the simply return {'cost':0, 'comment':'There is no new damage found.'} and no cost. Please make sure the response should be 100 percent in the json format any there is no any text before and after/end.",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": images_link[0],
                },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": images_link[1],
                },
                }
                
            ],
            }
        ],
        max_tokens=300,
        )

        data_str=response.choices[0].message.content
        data_str = data_str.replace("'", '"')
        try:
            data_json = json.loads(data_str)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=e)
        return data_json