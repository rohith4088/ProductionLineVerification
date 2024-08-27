# import json
# import requests

# API_TOKEN = "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"

# def query(payload='', parameters=None, options={'use_cache': False}):
#     API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
#     headers = {
#         "Authorization": f"Bearer {API_TOKEN}",
#         "Content-Type": "application/json"  # Add this line
#     }
#     body = {"inputs": payload, 'parameters': parameters, 'options': options}
#     response = requests.request("POST", API_URL, headers=headers, json=body)  # Use json parameter instead of data
    
#     print(f"Status Code: {response.status_code}")
#     print(f"Response Content: {response.text}")
    
#     try:
#         response.raise_for_status()
#         return response.json()[0]['generated_text']
#     except requests.exceptions.HTTPError as http_err:
#         return f"HTTP error occurred: {http_err}"
#     except json.JSONDecodeError as json_err:
#         return f"JSON decode error: {json_err}"
#     except Exception as err:
#         return f"An error occurred: {err}"

# parameters = {
#     'max_new_tokens': 25,  # number of generated tokens
#     'temperature': 0.5,   # controlling the randomness of generations
#     'end_sequence': "###" # stopping sequence for generation
# }

# prompt = "tell me about yourself"  # few-shot prompt
# data = query(prompt, parameters)
# print(data)

import json
import requests
import base64

API_TOKEN = "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"

def query(image_path, prompt=''):
    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare the payload
    payload = {
        "inputs": image_data,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
        }
    }
    
    # Add prompt if provided
    if prompt:
        payload["parameters"]["prompt"] = prompt
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")
    
    try:
        response.raise_for_status()
        return response.json()[0]['generated_text']
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except json.JSONDecodeError as json_err:
        return f"JSON decode error: {json_err}"
    except Exception as err:
        return f"An error occurred: {err}"

# Example usage
image_path = "resources/bw1.jpeg"  # Replace with your image path
prompt = "Describe the main objects and colors in this image:"  # Optional prompt

data = query(image_path, prompt)
print("Generated caption:", data)