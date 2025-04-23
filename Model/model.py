############################################################# Base Models #############################################################

from pydantic import BaseModel

class InputLicencePlate(BaseModel):
    image_url:str

class InputCarDamage(BaseModel):
    s3_url:str
    extension:str

class InputCarDamage2(BaseModel):
    s3_url:str
    extension:str
    img_type: str

class InputNightImage(BaseModel):
    image_url:str

class InputVllm(BaseModel):
    image_url:str