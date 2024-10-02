from Utils.licence_plate_util import extract_license_plate_number
from fastapi import HTTPException, Depends
from Auth.ApiAuthentication import get_api_token
from Services.s3_service import *
from fastapi import APIRouter
from Model.model import *
from Utils.util import *
from Utils.vehicle_damage_util import *



router=APIRouter()

######### End point for Number plate/licence plate# extraction ########
@router.post("/license-plate-number-extraction/")
async def number_plate_extraction(image_data:InputLicencePlate, api_token:str=Depends(get_api_token)):
    image_url = image_data.image_url
    if not image_url:
        raise HTTPException(status_code=400, detail="Image URL is missing in the request body")

    try:
        status,plate_number,detail = extract_license_plate_number(image_url)
        return {"status":status,"plateNumber":plate_number,"detail":detail}
    except HTTPException as exc:
        raise HTTPException(status_code=500, detail="Backend Issue.")


@router.post("/damage-detection/")
async def damage_detection(input: InputCarDamage, api_token: str = Depends(get_api_token)):
    dir_name = "s3_files/"
    s3_url = input.s3_url
    extension = input.extension
    file_name = random_name_generator()
    if s3_file_downloader(s3_url, file_name + extension):
        processed_file_path = None
        if extension.lower() == ".jpg":
            processed_file_path, message, damage_rectangle, original_img_info = damage_detection_in_image2(dir_name, file_name, extension)
            if processed_file_path is not None:
                uploaded_s3_link = upload_file(processed_file_path, file_name, extension)
            return {
                "image_s3_link": s3_url,
                "processed_img_s3_link": uploaded_s3_link,
                "extension": extension,
                "message": message,
                "org_img_info": original_img_info,
                "damages": damage_rectangle
            }
        elif extension.lower() == ".mp4":
            processed_file_path, message, unique_counts = damage_detection_in_video(dir_name, file_name, extension)
            if processed_file_path is not None:
                uploaded_s3_link = upload_file(processed_file_path, file_name, extension)
            return {
                "processed_file_path": file_name + extension,
                "uploaded_s3_link": uploaded_s3_link,
                "extension": extension,
                "message": message,
                "unique_counts": unique_counts
            }
    raise HTTPException(status_code=400, detail="Failed to process file")

######### End point for damage detection image/video extraction ########
@router.post("/damage-detection2/")
async def damage_detection(input:InputCarDamage, api_token:str=Depends(get_api_token)):
    dir_name="s3_files/"
    s3_url=input.s3_url
    extension=input.extension
    file_name=random_name_generator()

    if s3_file_downloader(s3_url,file_name+extension):
        processed_file_path=None
        if extension.lower()==".jpg":
            processed_file_path,message,damage_rectangle,original_img_info=damage_detection_in_image2(dir_name,file_name,extension)
            if processed_file_path!=None:
                uploaded_s3_link=upload_file(processed_file_path,file_name,extension)
            return {"image_s3_link":s3_url,"processed_img_s3_link":uploaded_s3_link,"extension":extension,"message":message,"org_img_info":original_img_info,"damages":damage_rectangle}
        