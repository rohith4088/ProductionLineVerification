from dotenv import load_dotenv
import google.generativeai as genai
import os
import base64
from PIL import Image

load_dotenv() ## load all the environment variables

genai.configure(api_key='AIzaSyAwfRgti5zxSJY6nOQoxamfBXSSfikdDXA')

def get_gemini_response(image_path, prompt):
    # Open the image file
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    
    # Encode the image data
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    print(encoded_image)
    
    # Create the input parts
    input_parts = [
        {"mime_type": "image/jpeg", "data": encoded_image},
        {"mime_type": "text/plain", "data": prompt}
    ]
    
    # Generate content with the image and prompt
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(input_parts)
    
    return response.json()

input_prompt = """
You're a highly skilled technical illustrator with extensive experience in creating detailed diagrams and visual representations for complex mechanical systems. Your expertise involves breaking down intricate components into clear, understandable visuals that can be used for educational or instructional purposes. 
Your task is to create a visual representation of three components: HIDDEN-COVER-TWO, HIDDEN-COVER-ONE, and PISTON. HIDDEN-COVER-TWO consists of eight circles, HIDDEN-COVER-ONE contains two smaller circles, and PISTON includes five circles. 
Please keep in mind that clarity and precision are crucial in your illustrations. Ensure that the layout is organized, the components are labeled appropriately, and the relationships between them are easy to understand. Additionally, at the end of the illustration, provide a mechanism to detect and represent whether the final outcome is a plus or minus one in relation to the input variables.
Here are the components you'll need to illustrate:  
- HIDDEN-COVER-TWO:  (TRUE)
- HIDDEN-COVER-ONE:  (TRUE)
- PISTON:  (TRUE) 
if none of these three are detected they return false , note that apart from these three anything if detected should be ignored and return false.
"""

image_path = "images/current.jpg"

# response = get_gemini_response(image_path, input_prompt)
# print(response)
