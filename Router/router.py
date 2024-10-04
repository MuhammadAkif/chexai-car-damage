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
                    "processed_file_path": processed_file_path,
                    "uploaded_s3_link": uploaded_s3_link,
                    "extension": extension,
                    "message": message,
                    "unique_counts": unique_counts
                }
    raise HTTPException(status_code=400, detail="Failed to process file")

######### End point for damage detection image/video extraction ########

@router.post("/damage-detection2/")
async def damage_detection(input: InputCarDamage, api_token: str = Depends(get_api_token)):
    dir_name = "s3_files/"
    s3_url = input.s3_url
    extension = input.extension
    file_name = random_name_generator()

    try:
        if s3_file_downloader(s3_url, file_name + extension):
            processed_file_path = None
            uploaded_s3_link = None
            message = ""
            damage_rectangle = []
            original_img_info = {}

            if extension.lower() == ".jpg":
                processed_file_path, message, damage_rectangle, original_img_info = damage_detection_in_image2(dir_name, file_name, extension)
                
                if processed_file_path is not None:
                    uploaded_s3_link = upload_file(processed_file_path, file_name, extension)
                else:
                    message = "No vehicle detected" if not message else message

            # Clean up the downloaded file
            if os.path.exists(dir_name + file_name + extension):
                os.remove(dir_name + file_name + extension)
            
            # Clean up the processed file if it exists
            if processed_file_path and os.path.exists(processed_file_path):
                os.remove(processed_file_path)

            return {
                "image_s3_link": s3_url,
                "processed_img_s3_link": uploaded_s3_link,
                "extension": extension,
                "message": message,
                "org_img_info": original_img_info,
                "damages": damage_rectangle
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to download file from S3")
    except Exception as e:
        # Log the error
        print(f"Error in damage detection: {str(e)}")
        # You might want to log this error to a file or error tracking service
        raise HTTPException(status_code=500, detail="An error occurred during damage detection")