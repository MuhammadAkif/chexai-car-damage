# from Utils.licence_plate_util import extract_license_plate_number
# from Utils.night_image_captured import is_night_captured_image
# from Utils.vllm import get_mistral_analysis
from fastapi import HTTPException, Depends
from Auth.ApiAuthentication import get_api_token
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from Services.s3_service import *
from fastapi import APIRouter
from Model.model import *
from Utils.util import *
from Utils.vehicle_damage_util import *
# from Utils.odometer import *
from Utils.vehicle_damage_three_level import *




router=APIRouter()

# ######### End point for Number plate/licence plate# extraction ########
# @router.post("/license-plate-number-extraction/")
# async def number_plate_extraction(image_data:InputLicencePlate, api_token:str=Depends(get_api_token)):
#     image_url = image_data.image_url
#     if not image_url:
#         raise HTTPException(status_code=400, detail="Image URL is missing in the request body")

#     try:
#         status,plate_number,detail = extract_license_plate_number(image_url)
#         return {"status":status,"plateNumber":plate_number,"detail":detail}
#     except HTTPException as exc:
#         raise HTTPException(status_code=500, detail="Backend Issue.")

# @router.post("/mileage-extraction/")
# async def extraction_milage(image_data:InputLicencePlate, api_token:str=Depends(get_api_token)):
#     image_url = image_data.image_url
#     if not image_url:
#         raise HTTPException(status_code=400, detail="Image URL is missing in the request body")

#     try:
#         status,mileage,detail = extract_mileage(image_url)
#         return {"status":status,"mileage":mileage,"detail":detail}
#     except HTTPException as exc:
#         raise HTTPException(status_code=500, detail="Backend Issue.")



# @router.post("/night_image_check/")
# async def is_night_image(image_data:InputNightImage, api_token:str=Depends(get_api_token)):
#     image_url = image_data.image_url
#     if not image_url:
#         raise HTTPException(status_code=400, detail="Image URL is missing in the request body")

#     try:
#         status= is_night_captured_image(image_url)
#         return {"status":status}
#     except HTTPException as exc:
#         raise HTTPException(status_code=500, detail="Backend Issue.")

# @router.post("/vllm/")
# async def vllm_analysis(image_data:InputNightImage, api_token:str=Depends(get_api_token)):
#     image_url = image_data.image_url
#     if not image_url:
#         raise HTTPException(status_code=400, detail="Image URL is missing in the request body")

#     try:
#         return get_mistral_analysis(image_url)
#     except HTTPException as exc:
#         raise HTTPException(status_code=500, detail="Backend Issue.")



# @router.post("/damage-detection/")
# async def damage_detection(input: InputCarDamage, api_token: str = Depends(get_api_token)):
#     dir_name = "s3_files/"
#     s3_url = input.s3_url
#     extension = input.extension
#     file_name = random_name_generator()
#     if s3_file_downloader(s3_url, file_name + extension):
#         processed_file_path = None
#         if extension.lower() == ".jpg":
#             processed_file_path, message, damage_rectangle, original_img_info = damage_detection_in_image2(dir_name, file_name, extension)
#             if processed_file_path is not None:
#                 uploaded_s3_link = upload_file(processed_file_path, file_name, extension)
#             return {
#                 "image_s3_link": s3_url,
#                 "processed_img_s3_link": uploaded_s3_link,
#                 "extension": extension,
#                 "message": message,
#                 "org_img_info": original_img_info,
#                 "damages": damage_rectangle
#             }
#         elif extension.lower() == ".mp4":
#             processed_file_path, message, unique_counts = damage_detection_in_video(dir_name, file_name, extension)
#             if processed_file_path is not None:
#                 uploaded_s3_link = upload_file(processed_file_path, file_name, extension)
#                 return {
#                     "processed_file_path": processed_file_path,
#                     "uploaded_s3_link": uploaded_s3_link,
#                     "extension": extension,
#                     "message": message,
#                     "unique_counts": unique_counts
#                 }
#     raise HTTPException(status_code=400, detail="Failed to process file")

# ######### End point for damage detection image/video extraction ########

# @router.post("/damage-detection2/")
# async def damage_detection(input: InputCarDamage, api_token: str = Depends(get_api_token)):
#     dir_name = "s3_files/"
#     s3_url = input.s3_url
#     extension = input.extension
#     file_name = random_name_generator()

#     try:
#         if s3_file_downloader(s3_url, file_name + extension):
#             processed_file_path = None
#             uploaded_s3_link = None
#             message = ""
#             damage_rectangle = []
#             original_img_info = {}

#             if extension.lower() == ".jpg":
#                 processed_file_path, message, damage_rectangle, original_img_info,final_status = damage_detection_in_image2(dir_name, file_name, extension)
                
#                 if processed_file_path is not None:
#                     uploaded_s3_link = upload_file(processed_file_path, file_name, extension)
#                 else:
#                     message = "Vehicle Not detected" if not message else message
#                     uploaded_s3_link = s3_url  # Use the original S3 URL if no processing occurred

#             # Clean up the downloaded file
#             if os.path.exists(dir_name + file_name + extension):
#                 os.remove(dir_name + file_name + extension)
            
#             # Clean up the processed file if it exists
#             if processed_file_path and os.path.exists(processed_file_path):
#                 os.remove(processed_file_path)

#             if final_status:
#                 final_status='fail'
#             else:
#                 final_status='pass'
#             response = {
#                 "image_s3_link": s3_url,
#                 "processed_img_s3_link": uploaded_s3_link or s3_url,  # Use original URL if uploaded_s3_link is None
#                 "extension": extension,
#                 "message": message,
#                 "final_status":final_status,
#                 "org_img_info": original_img_info,
#                 "damages": damage_rectangle
#             }

#             return response
#         else:
#             raise HTTPException(status_code=400, detail="Failed to download file from S3")
#     except Exception as e:
#         # Log the error
#         print(f"Error in damage detection: {str(e)}")
#         # You might want to log this error to a file or error tracking service
#         raise HTTPException(status_code=500, detail="An error occurred during damage detection")



# @router.post("/damage-detection3/")
# async def damage_detection3(input: InputCarDamage, api_token: str = Depends(get_api_token)):
#     dir_name = "s3_files/"
#     s3_url = input.s3_url
#     extension = input.extension
#     file_name = random_name_generator()

#     try:
#         if s3_file_downloader(s3_url, file_name + extension):
#             processed_file_path = None
#             uploaded_s3_link = None
#             message = ""
#             # Run the full damage detection pipeline
#             annotated_image, report = full_damage_detection(dir_name, file_name, extension, input.img_type)
#             # cv2.imwrite("processed_image.jpg", annotated_image)
            
#             if processed_file_path is not None:
#                 uploaded_s3_link = upload_file(processed_file_path, file_name, extension)
#             else:
#                 message = "Vehicle Not detected" if not message else message
#                 uploaded_s3_link = s3_url  # fallback to original S3 URL if processing did not occur

#             if os.path.exists(dir_name + file_name + extension):
#                 os.remove(dir_name + file_name + extension)
#             if processed_file_path and os.path.exists(processed_file_path):
#                 os.remove(processed_file_path)
            
#             final_status = report.get("final_status", "pass")
#             response = {
#                 "image_s3_link": s3_url,
#                 "processed_img_s3_link": uploaded_s3_link or s3_url,
#                 "extension": extension,
#                 "message": report.get("message", ""),
#                 "final_status": final_status,
#                 "org_img_info": report.get("org_img_info", {}),
#                 "damages": report.get("damages", []),
#                 "missing_body_parts": report.get("missing_body_parts", [])
#             }
#             # Use FastAPI's jsonable_encoder with custom_encoder to convert any numpy types
#             response = jsonable_encoder(
#                 response,
#                 custom_encoder={
#                     np.int64: int,
#                     np.int32: int,
#                     np.float64: float,
#                     np.float32: float
#                 }
#             )
#             return JSONResponse(content=response)
#         else:
#             raise HTTPException(status_code=400, detail="Failed to download file from S3")
#     except Exception as e:
#         print(f"Error in damage detection: {str(e)}")
#         raise HTTPException(status_code=500, detail="An error occurred during damage detection")

@router.post("/damage-detection3/")
async def damage_detection3(input: InputCarDamage, api_token: str = Depends(get_api_token)):
    dir_name = "s3_files/"
    s3_url = input.s3_url
    extension = input.extension
    file_name = random_name_generator()

    try:
        if s3_file_downloader(s3_url, file_name + extension):
            processed_file_path = None
            uploaded_s3_link = None
            message = ""

            # Run the full damage detection pipeline
            annotated_image, report = full_damage_detection(
                dir_name, file_name, extension, input.img_type)


            # Save the annotated image to a file
            processed_file_name = "processed_" + file_name
            processed_file_path = dir_name + processed_file_name + extension
            cv2.imwrite(processed_file_path, annotated_image)

            if processed_file_path is not None and "error" not in report:
                uploaded_s3_link = upload_file(
                    processed_file_path, processed_file_name, extension)
                final_status = report.get("final_status", "pass")
                message = report.get("message", "")
            else:
                message = report.get("message", "") or "Vehicle not detected"
                final_status = report.get("final_status", "fail")
                uploaded_s3_link = s3_url  # fallback to original S3 URL if processing did not occur

            # Clean up
            original_file_path = dir_name + file_name + extension
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            if processed_file_path and os.path.exists(processed_file_path):
                os.remove(processed_file_path)

            # final_status = report.get("final_status", "pass")
            response = {
                "image_s3_link": s3_url,
                "processed_img_s3_link": uploaded_s3_link or s3_url,
                "extension": extension,
                "message": message,
                "final_status": final_status,
                "org_img_info": report.get("org_img_info", {}),
                "damages": report.get("damages", []),
                "missing_body_parts": report.get("missing_body_parts", [])
            }

            # Encode numpy types
            response = jsonable_encoder(
                response,
                custom_encoder={
                    np.int64: int,
                    np.int32: int,
                    np.float64: float,
                    np.float32: float
                }
            )
            return JSONResponse(content=response)
        else:
            raise HTTPException(
                status_code=400, detail="Failed to download file from S3")
    except Exception as e:
        print(f"Error in damage detection: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred during damage detection")
