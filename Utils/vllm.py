import os
from mistralai import Mistral
from dotenv.main import load_dotenv
import json
import re
load_dotenv()

model = os.environ['MISTRALAI_MODEL']
mistral_client = Mistral(api_key=os.environ['MISTRALAI_KEY'])

def parse_json_response(input_data):
    try:
        if isinstance(input_data, str):
            # Use regex to find the JSON structure between curly braces
            json_match = re.search(r"\{[\s\S]*\}", input_data)
            if json_match:
                # Extract the JSON portion
                input_data = json_match.group(0)
            else:
                return {"error": "No JSON found in input data"}
            
            parsed_data = json.loads(input_data)
        elif isinstance(input_data, dict):
            parsed_data = input_data
        else:
            raise ValueError("Unsupported input format")

        return parsed_data
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}





def get_mistral_analysis(s3_url):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":"""
                        Detect damages from this vehicle image and return a response of any small dents and scratches.
                        response should be like this {"final_status":pass/fail, "message":"short description only"}."""
                },
                {
                    "type": "image_url",
                    "image_url": s3_url
                }
            ]
        }
    ]
    try:
        chat_response = mistral_client.chat.complete(model=model, messages=messages)
        # print("Chat response vllm: ", chat_response.choices[0].message.content)
        parsed_respone=parse_json_response(chat_response.choices[0].message.content)
        # print("Chat response after parsing: ", parsed_respone)
        return parsed_respone    
    except Exception as e:
        return f"Error getting Mistral analysis: {str(e)}"
    

