from botocore.exceptions import NoCredentialsError
from dotenv.main import load_dotenv
import boto3
import os

load_dotenv()

s3_bucket = boto3.client('s3', aws_access_key_id=os.environ['AWS_S3_ACCESS_KEY_ID'],
                      aws_secret_access_key=os.environ['AWS_S3_SECRET_ACCESS_KEY'])


def upload_file_to_s3_bucket(file_path, s3_file):
    s3_file_path="car_damage_detection/"+s3_file
    try:
        s3_bucket.upload_file(file_path, os.environ['AWS_S3_BUCKET_NAME'], s3_file_path)

        s3_url = f"https://{os.environ['AWS_S3_BUCKET_NAME']}.s3.amazonaws.com/{s3_file_path}"
        return s3_url

    except Exception as e:
        print(f'An error occurred: {e}')
        return None
    except NoCredentialsError:
        return None