from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
import os

api_token_header=APIKeyHeader(name="api_token",auto_error=True)
APIKeyHeader(name="api_token",auto_error=True)
async def get_api_token(api_token: str =Depends(api_token_header)):
    if api_token!=os.environ['API_TOKEN']:
        raise HTTPException(status_code=400,detail="Invalid API Token.")
    return api_token