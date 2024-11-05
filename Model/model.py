############################################################# Base Models #############################################################

from pydantic import BaseModel

class InputLicencePlate(BaseModel):
    image_url:str

class InputCarDamage(BaseModel):
    s3_url:str
    extension:str

class InputNightImage(BaseModel):
    image_url:str